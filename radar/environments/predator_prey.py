import random
import numpy
from gym import spaces
from radar.environments.environment import GridWorldEnvironment, GridWorldObject, GRIDWORLD_ACTIONS
from radar.utils import get_param_or_default, check_value_not_none, get_value_if
from settings import params
# from radar.experiments import
from radar.agents.controller import Controller

AGENT_CHANNEL = 0
PREY_CHANNEL = 1
OBSTACLE_CHANNEL = 2
ADVERSARY_CHANNEL = 3

PREDATOR_PREY_CHANNELS = [AGENT_CHANNEL, PREY_CHANNEL, OBSTACLE_CHANNEL]


class Predator(GridWorldObject):

    def __init__(self, id, initial_position, env, fixed_initial_position=False):
        super(Predator, self).__init__(id, initial_position, env, fixed_initial_position)
        self.capture_participation = 0.0

    # noinspection PyAttributeOutsideInit
    def reset(self):
        super().reset()
        self.capture_participation = 0.0

    def add_capture_participation(self, participation):
        self.capture_participation += participation


class Prey(GridWorldObject):
    def __init__(self, id, initial_position, env, fixed_initial_position=False):
        super(Prey, self).__init__(id, initial_position, env, fixed_initial_position)
        self.params = params
        self.height = params["height"]
        self.width = params["width"]
        self.capture_participation = 0.0
        self.rect_path = self.Rect_path()
        self.po = 0
        self.point = 1


    def move_rect(self):
        x_0, y_0 = self.position
        i = self.get_path_i(self.rect_path, x_0, y_0)
        if i == (2*self.width+2*(self.height -2)-1):
            i = -1
        return self.set_position(self.rect_path[i+1])

    def move_str(self, path, position):
        x_0, y_0 = position
        i = self.get_path_i(path, x_0, y_0)
        if i == self.width - 1:
            self.point = -self.point  # 翻
            return self.set_position(path[i - 1])
        if i < self.width - 1:
            return self.set_position(path[i + 1])

    def get_path_i(self, path, x_0, y_0):
        for i, (x, y) in enumerate(path):
            if x == x_0 and y == y_0:
                return i

    def set_position(self, new_position):
        # obstacle collision check
        if new_position in self.env.obstacles:
            return
        self.position = new_position

    def Rect_path(self):
        prey_path = []
        for i in range(self.width):
            prey_path.append((i, 0))
        for j in range(1, self.height):
            prey_path.append((self.width - 1, j))
        for i in range(1, self.width):
            prey_path.append((self.width - 1 - i, self.height - 1))
        for j in range(1, self.height - 1):
            prey_path.append((0, self.height - 1 - j))
        return prey_path

    def Straight_path(self, position):
        h_path = []
        v_path = []
        x_0, y_0 = position
        for i in range(self.width):
            v_path.append((i,y_0))
        for j in range(self.height):
            h_path.append((x_0,j))
        return h_path,v_path


class VehiclePursuitEnvironment(GridWorldEnvironment):
    def __init__(self, params):
        super(VehiclePursuitEnvironment, self).__init__(params)
        self.nr_channels_global = len(PREDATOR_PREY_CHANNELS)
        # self.nr_channels_local = self.nr_channels_global + 2
        self.params = params
        self.nr_preys = int(params["nr_agents"] / 2)
        self.nr_capturers_required = 2
        self.failed_capture_penalty = get_param_or_default(params, "failed_capture_penalty", 0)
        self.agents = [Predator(i, None, self, False) for i in range(params["nr_agents"])]
        self.preys = [Prey(i, None, self, False) for i in range(self.nr_preys)]
        # self.preys = [GridWorldObject(i, None, self, False) for i in range(self.nr_preys)]
        self.global_observation_space = spaces.Box(-numpy.inf, numpy.inf,
                                                   shape=(self.nr_channels_global, self.width, self.height))
        self.view_range = get_param_or_default(params, "view_range", 5)
        default_capture_distance = 2
        self.capture_distance = get_param_or_default(params, "capture_distance", default_capture_distance)
        self.prey_capture_count = 0
        self.failed_capture_count = 0
        if params["local_observation_format"] == 0:
            self.nr_channels_local = self.nr_agents + 2
            self.local_observation_space = spaces.Box(
                -numpy.inf, numpy.inf, shape=(self.nr_channels_local, self.view_range, self.view_range))
        elif params["local_observation_format"] == 1:
            self.nr_channels_local = 2
            self.local_observation_space = spaces.Box(
                -numpy.inf, numpy.inf, shape=(self.nr_channels_local, self.width, self.height))
        else:
            self.nr_channels_local = 3
            self.local_observation_space = spaces.Box(
                -numpy.inf, numpy.inf, shape=(self.nr_channels_local, self.width, self.height))

    def global_state(self):
        state = numpy.zeros(self.global_observation_space.shape)
        for agent in self.agents:
            x, y = agent.position
            state[AGENT_CHANNEL][x][y] += 1

        for prey in self.preys:
            x, y = prey.position
            state[PREY_CHANNEL][x][y] += 1
        for obstacle in self.obstacles:
            x, y = obstacle
            state[OBSTACLE_CHANNEL][x][y] += 1
        # if params["test"]:
        #     # print_state = numpy.ones(self.global_observation_space.shape[1:])*8
        #     print_state = numpy.array([[" " for _ in range(13)] for _ in range(13)])
        #     index = 0
        #     for agent in self.agents:
        #         x, y = agent.position
        #         print_state[x][y] = index
        #         index += 1
        #
        #     for prey in self.preys:
        #         x, y = prey.position
        #         print_state[x][y] = 9
        #     # for obstacle in self.obstacles:
        #     #     x, y = obstacle
        #     #     print_state[x][y] += 3
        #     print(print_state)
        return state

    def local_observation(self, agent: object = None) -> object:
        if self.nr_agents == 1:
            agent = self.agents[0]
        observation = numpy.zeros(self.local_observation_space.shape)
        if agent.done:
            return observation
        x_0, y_0 = agent.position
        x_center = int(self.view_range / 2)
        y_center = int(self.view_range / 2)
        visible_positions = [(x, y) for x in range(-x_center + x_0, x_center + 1 + x_0) for y in
                             range(-y_center + y_0, y_center + 1 + y_0)]

        if self.params["local_observation_format"] == 0:
            for visible_position in visible_positions:
                x, y = visible_position
                if x < 0 or y < 0 or x >= self.width or y >= self.height or (x, y) in self.obstacles:
                    dx, dy = self.relative_position(agent, visible_position)
                    observation[self.nr_channels_local - 1][x_center + dx][y_center + dy] = 1

            observation[0][x_center][y_center] += 1

            for index, agent in enumerate(self.agents):
                junction_vis = self.junction_observation(agent)
                for prey in self.preys:
                    if prey.position in junction_vis:
                        dx, dy = self.relative_position(agent, prey.position)
                        observation[PREY_CHANNEL + index][x_center + dx][y_center + dy] += 1

        elif self.params["local_observation_format"] == 1:
            # 第一层：5*5内的障碍物信息
            for visible_position in visible_positions:
                x, y = visible_position
                if x < 0 or y < 0 or x >= self.width or y >= self.height or (x, y) in self.obstacles:
                    dx, dy = self.relative_position(agent, visible_position)
                    if (x_0 + dx, y_0 + dy) in self.grid:
                        observation[0][x_0 + dx][y_0 + dy] = 1

            # 第二层：猎物的信息
            for index, agent in enumerate(self.agents):
                junction_vis = self.junction_observation(agent)
                x_1, y_1 = agent.position
                for prey in self.preys:
                    if prey.position in junction_vis:
                        dx, dy = self.relative_position(agent, prey.position)
                        observation[1][x_1 + dx][y_1 + dy] += 1

        else:
            # 第一层：自己的位置信息
            observation[0][x_0][y_0] += 1

            # 第二层：5*5内的障碍物信息
            for visible_position in visible_positions:
                x, y = visible_position
                if x < 0 or y < 0 or x >= self.width or y >= self.height or (x, y) in self.obstacles:
                    dx, dy = self.relative_position(agent, visible_position)
                    if (x_0 + dx, y_0 + dy) in self.grid:
                        observation[1][x_0 + dx][y_0 + dy] = 1

            # 第三层：猎物的信息
            for index, agent in enumerate(self.agents):
                junction_vis = self.junction_observation(agent)
                x_1, y_1 = agent.position
                for prey in self.preys:
                    if prey.position in junction_vis:
                        dx, dy = self.relative_position(agent, prey.position)
                        observation[2][x_1 + dx][y_1 + dy] += 1

        return observation

    def domain_statistic(self):
        return sum([agent.capture_participation for i, agent in enumerate(self.agents)])

    def step(self, joint_action, param):
        prey_choice = param
        self.time_step += 1
        rewards = numpy.zeros(self.nr_agents)
        agent_positions = []
        for i, agent, action in zip(range(self.nr_agents), self.agents, joint_action):
            if not agent.done:
                agent.move(action)
            agent_positions.append(agent.position)
        if prey_choice == 0:
            prey_actions = self.prey_policy_random()
            for i, agent, action in zip(range(self.nr_preys), self.preys, prey_actions):
                agent.move(action)

        elif prey_choice == 1:
            prey_actions = self.prey_policy_fixed()
            for i, agent, action in zip(range(self.nr_preys), self.preys, prey_actions):
                agent.move(action)

        elif prey_choice == 2:
            for i, agent in zip(range(self.nr_preys), self.preys):
                if self.if_in_rect(agent) == True:
                    agent.move_rect()
                elif self.if_in_rect(agent) == False:
                    action = self.prey_policy_rect(agent)
                    agent.move(action)

        else:
            for i, agent in zip(range(self.nr_preys), self.preys):
                h_path, v_path = agent.Straight_path(agent.position)
                x_0, y_0 = agent.position
                if x_0 < self.width - 1 and y_0 < self.height - 1:  # 除最后一行，最后一列
                    if (x_0 + 1, y_0) not in self.obstacles:  # 可以垂直，垂直走
                        if agent.po == 0:
                            v1_path = self.get_point(agent, v_path)  # 第12步没有翻转 #下一 步翻转
                            agent.move_str(v1_path, agent.position)
                        elif agent.po == 1:
                            agent.po = 1
                            h1_path = self.get_point(agent, h_path)
                            agent.move_str(h1_path, agent.position)
                    else:  # 水平
                        agent.po = 1
                        h1_path = self.get_point(agent, h_path)
                        agent.move_str(h1_path, agent.position)

                elif x_0 == self.width - 1:
                    if (x_0 - 1, y_0) not in self.obstacles:  # 在第12行，如果它上面没有，直走
                        if agent.po == 0:
                            v1_path = self.get_point(agent, v_path)
                            agent.move_str(v1_path, agent.position)
                        elif agent.po == 1:
                            h1_path = self.get_point(agent, h_path)
                            agent.move_str(h1_path, agent.position)
                            agent.po = 1
                    else:  # 水平
                        h1_path = self.get_point(agent, h_path)
                        agent.move_str(h1_path, agent.position)
                        agent.po = 1
                elif y_0 == self.height - 1:
                    if agent.po == 0:
                        v1_path = self.get_point(agent, v_path)
                        agent.move_str(v1_path, agent.position)
                    elif agent.po == 1:
                        h1_path = self.get_point(agent, h_path)
                        agent.move_str(h1_path, agent.position)
                        agent.po = 1

        for prey in self.preys:
            x_1, y_1 = prey.position
            capturers = []
            main_capturers = []
            for i, agent_position in enumerate(agent_positions):
                x_0, y_0 = agent_position
                distance = max(abs(x_1 - x_0), abs(y_1 - y_0))
                if distance <= self.capture_distance:
                    capturers.append(i)
                if prey.position == agent_position:
                    main_capturers.append(i)
            nr_capturers = 1.0 * len(capturers)
            if nr_capturers >= self.nr_capturers_required:
                for i in main_capturers:
                    participation = 1.0 / len(main_capturers)
                    # assert participation < 1, "partiticipation was {}".format(participation)
                    rewards[i] += participation
                    self.agents[i].add_capture_participation(participation)
                    self.prey_capture_count += participation
                prey.po = 0
                prey.point = 1
                prey.reset()
        if self.time_step >= self.time_limit:
            for agent in self.agents:
                agent.done = True
        global_reward = sum(rewards)
        self.discounted_return += global_reward * (self.gamma ** (self.time_step - 1))
        self.undiscounted_return += global_reward
        assert len(self.preys) == self.nr_preys
        return self.joint_observation(), \
               rewards, [agent.done for agent in self.agents], {
                   "preys": [prey.position for prey in self.preys]
               }

    def reset(self):
        super().reset()
        self.obstacle_free_positions = [(x, y) for x in range(self.width) \
                                        for y in range(self.height) if (x, y) not in self.obstacles]
        for prey in self.preys:
            prey.reset()
        self.failed_capture_count = 0
        self.prey_capture_count = 0
        return self.joint_observation()

    def state_summary(self):
        summary = super(VehiclePursuitEnvironment, self).state_summary()
        summary["obstacles"] = self.obstacles
        summary["width"] = self.width
        summary["height"] = self.height
        summary["preys"] = [prey.state_summary() for prey in self.preys]
        summary["view_range"] = self.view_range
        summary["prey_capture_count"] = self.prey_capture_count
        summary["failed_capture_count"] = self.failed_capture_count
        return summary

    def prey_policy_random(self):
        joint_action = [random.choice(self.actions) \
                               for agent in self.preys]
        return joint_action

    def prey_policy_fixed(self):
        joint_action = [self.actions[0] for agent in self.preys]
        return joint_action

    def if_in_rect(self, agent):
        rect_path = agent.rect_path
        x, y = agent.position
        if (x, y) in rect_path:
            return True
        else:
            return False

    def get_point(self, agent, path):
        if agent.point == -1:
            path.reverse()
            return path
        else:
            return path

    def prey_policy_rect(self,agent):
        x_0, y_0 = agent.position
        if x_0 < (self.width-1) or y_0 < (self.height-1):
            if (x_0+1, y_0) not in self.obstacles:
                action = self.actions[4]
                return action
            elif (x_0, y_0+1) not in self.obstacles:
                action = self.actions[2]
                return action


    def junction_observation(self, agent):
        x_0, y_0 = agent.position
        x_center = int(self.view_range / 2)
        y_center = int(self.view_range / 2)
        visible_prey_positions = []
        scan_positions = []
        for x in range(-x_center + x_0, x_center + 1 + x_0):
            scan_positions.append((x, y_0))
            if (x, y_0) in self.obstacles:
                scan_positions = []
                break
        visible_prey_positions += scan_positions
        scan_positions = []
        for y in range(-y_center + y_0, y_center + 1 + y_0):
            scan_positions.append((x_0, y))
            if (x_0, y) in self.obstacles:
                scan_positions = []
                break
        visible_prey_positions += scan_positions
        return visible_prey_positions


PREDATOR_PREY_LAYOUTS = {
    # N = 4
    "VehiclePursuit-4": ("""
           . . . . . . . . . . .
           . # . # . # . # . # .
           . . . . . . . . . . .
           . # . # . # . # . # .
           . . . . . . . . . . .
           . # . # . # . # . # .
           . . . . . . . . . . .
           . # . # . # . # . # .
           . . . . . . . . . . .
           . # . # . # . # . # .
           . . . . . . . . . . . 
        """, 4),

    "VehiclePursuit-8": ("""
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . .
           . # . # . # . # . # . # .
           . . . . . . . . . . . . . 
        """, 8)

}

def make(domain, params):
    params["nr_actions"] = len(GRIDWORLD_ACTIONS)
    params["gamma"] = 0.95
    params["obstacles"] = []
    params["time_limit"] = 50
    params["fixed_initial_position"] = False
    params["collisions_allowed"] = True
    layout, params["nr_agents"] = PREDATOR_PREY_LAYOUTS[domain]
    layout = layout.splitlines()
    params["width"] = 0
    params["height"] = 0
    for _,line in enumerate(layout):
        splitted_line = line.strip().split()
        if splitted_line:
            for x,cell in enumerate(splitted_line):
                if cell == '#':
                    params["obstacles"].append((x,params["height"]))
                params["width"] = x
            params["height"] += 1
    params["width"] += 1

    if domain.startswith("VehiclePursuit-"):
        return VehiclePursuitEnvironment(params)


if __name__ == '__main__':
    parser = {}
    env = make("VehiclePursuit-8", parser)


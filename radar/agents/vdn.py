from radar.agents.dqn import DQNLearner
import torch
import torch.nn.functional as F


class VDNLearner(DQNLearner):

    def __init__(self, params):
        self.global_value_network = None
        self.global_target_network = None
        super(VDNLearner, self).__init__(params)
        self.zero_actions = torch.zeros(self.batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        if self.params["UPDeT"]:
            self.train_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().unsqueeze(0).expand(
                self.params["batch_size"], params["nr_agents"], 1, -1).cpu().numpy()

    def update(self, state, obs, joint_action, rewards, next_state, next_obs, dones):
        super(VDNLearner, self).update_transition(state, obs, joint_action, rewards, next_state, next_obs, dones)
        self.warmup_phase = max(0, self.warmup_phase - 1)
        if self.warmup_phase <= 0:
            self.training_count += 1
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            if self.params["UPDeT"]:
                minibatch = self.memory.sample_time_batch(self.batch_size, max_time_length=self.params["max_history_length"])
            else:
                minibatch = self.memory.sample_batch(self.batch_size)
            nr_protagonists = self.nr_agents
            self.train_step(minibatch, nr_protagonists)
            self.update_target_network()
            return True
        return False

    def global_value(self, network, Q_values, states):
        if self.params["UPDeT"]:
            return Q_values.sum(-1).view(-1)
        else:
            return Q_values.sum(1)

    def train_step(self, minibatch, nr_agents):
        minibatch_data = self.collect_minibatch_data(minibatch)
        return self.train_step_with(minibatch_data, nr_agents=nr_agents)

    def train_step_with(self, minibatch_data, target_values=None, nr_agents=None):
        states = minibatch_data["states"]
        next_states = minibatch_data["next_states"]
        histories = minibatch_data["pro_histories"]
        actions = minibatch_data["pro_actions"]
        next_histories = minibatch_data["next_pro_histories"]
        rewards = minibatch_data["pro_rewards"]
        optimizer = self.protagonist_optimizer
        if nr_agents is None:
            nr_agents = self.nr_agents
        if self.params["UPDeT"]:
            next_histories = next_histories.squeeze(dim=2)
            histories = histories.squeeze(dim=2)
            self.train_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().unsqueeze(0).expand(
                self.params["batch_size"], self.params["nr_agents"], 1, -1).cpu().numpy()
            time_agent_out = []
            for t in range(self.params["max_history_length"]):
                agent_out, h = self.policy_net(histories[:, t],
                                               torch.tensor(self.train_hidden_state).reshape(self.params["batch_size"],
                                                                                             -1, 1,
                                                                                             self.params["emb"]).to(
                                                   self.device))
                self.train_hidden_state = h.cpu().detach().numpy()
                agent_out = agent_out.view(self.params["batch_size"], self.params["nr_agents"], -1)
                time_agent_out.append(agent_out)
            time_agent_out = torch.stack(time_agent_out, dim=1)
            actions = actions.view(self.params["batch_size"], -1, self.params["nr_agents"]).unsqueeze(dim=-1).type(torch.int64)
            chosen_action_qvals = torch.gather(time_agent_out, dim=3, index=actions).squeeze(3)

            target_time_agent_out = []
            self.train_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().unsqueeze(0).expand(
                self.params["batch_size"], self.params["nr_agents"], 1, -1).cpu().numpy()
            for t in range(self.params["max_history_length"]):
                agent_out, h = self.target_net(histories[:, t],
                                               torch.tensor(self.train_hidden_state).reshape(self.params["batch_size"],
                                                                                             -1, 1,
                                                                                             self.params["emb"]).to(
                                                   self.device))
                self.train_hidden_state = h.cpu().detach().numpy()
                agent_out = agent_out.view(self.params["batch_size"], self.params["nr_agents"], -1)
                target_time_agent_out.append(agent_out)
            target_time_agent_out = torch.stack(target_time_agent_out, dim=1)
            target_max_qvals = target_time_agent_out.max(dim=3)[0]

            # todo 将无法执行的动作给予-9999999的target Q值

            # chosen_action_qvals = chosen_action_qvals.view(-1, nr_agents)
            chosen_action_qvals = self.global_value(self.global_value_network, chosen_action_qvals, states)
            target_max_qvals = self.global_value(self.global_target_network, target_max_qvals, next_states)


            rewards = rewards.view(self.params["batch_size"], self.params["max_history_length"], -1)
            rewards = torch.sum(rewards, dim=2)/ self.params["nr_agents"]
            rewards = rewards.view(-1)
            targets = rewards + self.gamma * target_max_qvals

            optimizer.zero_grad()
            loss = F.mse_loss(chosen_action_qvals, targets)
            loss.backward()
            optimizer.step()

            # histories = histories.squeeze(1)
            # state_action_values, _ = self.policy_net(histories, torch.tensor(self.target_hidden_state).reshape(
            #     self.params["batch_size"], -1, 1, self.params["emb"]).to(self.device))
            # state_action_values = state_action_values.gather(1, actions.unsqueeze(1)).squeeze()
        else:
            state_action_values = self.policy_net(histories).gather(1, actions.unsqueeze(1)).squeeze()

            state_action_values = state_action_values.view(-1, nr_agents)
            state_action_values = self.global_value(self.global_value_network, state_action_values, states)
            if target_values is None:
                rewards = rewards.view(-1, nr_agents)
                assert rewards.size(0) == states.size(0)
                rewards = rewards.gather(1, self.zero_actions).squeeze()
                if self.params["UPDeT"]:
                    next_histories = next_histories.squeeze(1)
                    next_state_values, _ = self.target_net(next_histories,
                                                           torch.tensor(self.target_hidden_state).reshape(
                                                               self.params["batch_size"], -1, 1, self.params["emb"]).to(
                                                               self.device))
                    next_state_values = next_state_values.max(1)[0]
                    next_state_values = next_state_values.view(-1, nr_agents)
                else:
                    next_state_values = self.target_net(next_histories).max(1)[0]
                    next_state_values = next_state_values.view(-1, nr_agents)
                next_state_values = self.global_value(self.global_target_network, next_state_values,
                                                      next_states).detach()
                target_values = rewards + self.gamma * next_state_values
            optimizer.zero_grad()
            loss = F.mse_loss(state_action_values, target_values)
            loss.backward()
            optimizer.step()
        return True

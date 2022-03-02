from T3OMVP.agents.controller import DeepLearningController, ReplayMemory
from T3OMVP.modules import MLP, AdversarialModule, UPDeT
from T3OMVP.utils import argmax, get_param_or_default
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class DQNNet(torch.nn.Module):
    def __init__(self, input_shape, outputs, max_history_length):
        super(DQNNet, self).__init__()
        self.fc_net = MLP(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, outputs)

    def forward(self, x):
        x = self.fc_net(x)
        return self.action_head(x)

class DQNUPDeT(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, device="cuda", params=None):
        super(DQNUPDeT, self).__init__()
        self.params = params
        self.fc_net = UPDeT(input_shape=input_shape, params=params)
        self.action_head = nn.Linear(nr_actions, nr_actions)
        # self.value_head = nn.Linear(nr_actions, 1)
        self.hidden_state = self.fc_net.init_hidden().expand(params["nr_agents"], 1, -1)

    def forward(self, x, h):
        x, _h = self.fc_net(x, h, self.params["nr_agents"], int(self.params["nr_agents"]/2))
        # return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
        return self.action_head(x), _h


class DQNLearner(DeepLearningController):

    def __init__(self, params):
        super(DQNLearner, self).__init__(params)
        self.epsilon = 1.0
        self.epsilon_decay = get_param_or_default(params, "epsilon_decay", 0.0001)
        self.epsilon_min = get_param_or_default(params, "epsilon_min", 0.01)
        self.batch_size = get_param_or_default(params, "batch_size", 64)
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        if params["UPDeT"]:
            network_constructor = lambda in_shape, actions, length: DQNUPDeT(input_shape, actions, length, self.device, self.params)
        else:
            network_constructor = lambda in_shape, actions, length: DQNNet(in_shape, actions, length)

        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(self.device)
        self.target_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(self.device)
        if params["UPDeT"]:
            self.hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().expand(1, params["nr_agents"], 1, -1).cpu().numpy()
            self.target_hidden_state = self.target_net.protagonist_net.fc_net.init_hidden().expand(self.params["batch_size"], params["nr_agents"], 1, -1).cpu().numpy()
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.update_target_network()

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):

        action_probs = []
        used_epsilon = self.epsilon_min
        if training_mode:
            used_epsilon = self.epsilon
        if agent_ids is None:
            agent_ids = self.agent_ids
        if self.params["UPDeT"]:
            if self.params["max_history_length"] > 1:
                histories = [histories[0]]
            team_history = [[joint_obs[i] for i in agent_ids] for joint_obs in histories]
            team_history = torch.tensor(team_history, device=self.device, dtype=torch.float32)
            Q_values, h = self.policy_net(team_history, torch.tensor(self.hidden_state).reshape(1, -1, 1, self.params["emb"]).to(self.device))
            self.hidden_state = h.cpu().detach().numpy()
            Q_values = Q_values.detach().cpu().numpy()
            for agent in range(len(Q_values)):
                probs = used_epsilon*numpy.ones(self.nr_actions)/self.nr_actions
                rest_prob = 1 - sum(probs)
                probs[argmax(Q_values[agent])] += rest_prob
                action_probs.append(probs / sum(probs))
            # print("11:{}".format(torch.cuda.memory_allocated(0)))
            return action_probs

        else:
            for i, agent_id in enumerate(agent_ids):
                # print("6:{}".format(torch.cuda.memory_allocated(0)))
                history = [[joint_obs[i]] for joint_obs in histories]
                # print("7:{}".format(torch.cuda.memory_allocated(0)))
                history = torch.tensor(history, device=self.device, dtype=torch.float32)
                # print("8:{}".format(torch.cuda.memory_allocated(0)))
                Q_values = self.policy_net(history).detach().cpu().numpy()
                # print("9:{}".format(torch.cuda.memory_allocated(0)))
                assert len(Q_values) == 1, "Expected length 1, but got shape {}".format(Q_values.shape)
                probs = used_epsilon*numpy.ones(self.nr_actions)/self.nr_actions
                rest_prob = 1 - sum(probs)
                probs[argmax(Q_values[0])] += rest_prob
                action_probs.append(probs/sum(probs))
            return action_probs

    def update(self, state, obs, joint_action, rewards, next_state, next_obs, dones):
        super(DQNLearner, self).update(state, obs, joint_action, rewards, next_state, next_obs, dones)
        if self.warmup_phase <= 0:
            minibatch = self.memory.sample_batch(self.batch_size)
            minibatch_data = self.collect_minibatch_data(minibatch)
            histories = minibatch_data["pro_histories"]
            next_histories = minibatch_data["next_pro_histories"]
            actions = minibatch_data["pro_actions"]
            rewards = minibatch_data["pro_rewards"]
            self.update_step(histories, next_histories, actions, rewards, self.protagonist_optimizer, False)
            self.update_target_network()
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            self.training_count += 1
            return True
        return False

    def update_step(self, histories, next_histories, actions, rewards, optimizer):
        Q_values = self.policy_net(histories).gather(1, actions.unsqueeze(1)).squeeze()
        next_Q_values = self.target_net(next_histories).max(1)[0].detach()
        target_Q_values = rewards + self.gamma*next_Q_values
        optimizer.zero_grad()
        loss = F.mse_loss(Q_values, target_Q_values)
        loss.backward()
        optimizer.step()
        return loss

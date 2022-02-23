import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from radar.utils import argmax, get_param_or_default
from radar.agents.controller import DeepLearningController, get_param_or_default
from radar.modules import MLP, AdversarialModule, MLP3D, TimeTransformer, UPDeT, TimeTransformerNew
import time

class PPONet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False):
        super(PPONet, self).__init__()
        self.fc_net = MLP(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        if q_values:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        else:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)

    def forward(self, x, use_gumbel_softmax=False):
        x = self.fc_net(x)
        if use_gumbel_softmax:
            return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)


class PPOHiddenNet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False):
        super(PPOHiddenNet, self).__init__()
        self.fc_net = MLP(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions+1)
        if q_values:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions+1)
        else:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)

    def forward(self, x, use_gumbel_softmax=False):
        x = self.fc_net(x)
        ah = self.action_head(x)
        if use_gumbel_softmax:
            return F.gumbel_softmax(self.action_head(x)[:, :-1], hard=True, dim=-1), self.value_head(x)
        return F.softmax(ah[:, :-1], dim=-1), self.value_head(x)[:, :-1], ah[:, -1]


class PPO3DNet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False):
        super(PPO3DNet, self).__init__()
        self.fc_net = MLP3D(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        if q_values:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        else:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)

    def forward(self, x, use_gumbel_softmax=False):
        x = self.fc_net(x)
        if use_gumbel_softmax:
            return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)


# class PPOTimeTransformer(nn.Module):
#     def __init__(self, input_shape, nr_actions, max_history_length, q_values=False, device="cuda", params=None):
#         super(PPOTimeTransformer, self).__init__()
#         # self.fc_net = TimeTransformer(input_shape, max_history_length, device=device)
#         self.fc_net = TimeTransformerNew(input_shape, max_history_length, params)
#         self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
#         if q_values:
#             self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
#         else:
#             self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)
#
#     def forward(self, x, use_gumbel_softmax=False):
#         x = self.fc_net(x)
#         if use_gumbel_softmax:
#             return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
#         return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

class PPOTimeTransformer(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False, device="cuda", params=None):
        super(PPOTimeTransformer, self).__init__()
        # self.fc_net = TimeTransformer(input_shape, max_history_length, device=device)
        self.fc_net = TimeTransformerNew(input_shape, max_history_length, params, q_values)


    def forward(self, x, use_gumbel_softmax=False):
        x, value = self.fc_net(x)

        return F.softmax(x, dim=-1), value


class PPOUPDeT(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, device="cuda", params=None):
        super(PPOUPDeT, self).__init__()
        self.params = params
        self.fc_net = UPDeT(input_shape=input_shape, params=params).to(device)
        self.action_head = nn.Linear(nr_actions, nr_actions)
        # self.value_head = nn.Linear(nr_actions, 1)
        self.hidden_state = self.fc_net.init_hidden().expand(params["nr_agents"], 1, -1)
        self.value_head = nn.Linear(nr_actions, 1)

    def forward(self, x, h):
            x, _h = self.fc_net(x, h, self.params["nr_agents"], int(self.params["nr_agents"] / 2))

            # return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
            return F.softmax(self.action_head(x), dim=-1), _h, self.value_head(x)


class PPOLearner(DeepLearningController):

    def __init__(self, params):
        super(PPOLearner, self).__init__(params)
        self.nr_epochs = get_param_or_default(params, "nr_epochs", 5)
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2)
        self.use_q_values = get_param_or_default(params, "use_q_values", False)
        self.batch_size = get_param_or_default(params, "batch_size", 64)
        self.epsilon_min = get_param_or_default(params, "epsilon_min", 0.01)
        if self.params["UPDeT"]:
            self.nr_episodes = self.batch_size - 1
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        if params["conv3d"]:
            network_constructor = lambda in_shape, actions, length: PPO3DNet(in_shape, actions, length, self.use_q_values)
        elif params["timeTransformer"]:
            network_constructor = lambda in_shape, actions, length: PPOTimeTransformer(in_shape, actions, length, self.use_q_values, self.device, self.params)
        elif params["UPDeT"]:
            network_constructor = lambda in_shape, actions, length: PPOUPDeT(input_shape, actions, length, self.device, self.params)
        else:
            network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, self.use_q_values)
        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(self.device)
        if params["UPDeT"]:
            self.hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().expand(1, params["nr_agents"], 1,
                                                                                            -1).cpu().numpy()
            self.target_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().expand(
                self.params["batch_size"], params["nr_agents"], 1, -1).cpu().numpy()
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)

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
            Q_values, h, _ = self.policy_net(team_history, torch.tensor(self.hidden_state).reshape(1, -1, 1, self.params["emb"]).to(self.device))
            self.hidden_state = h.cpu().detach().numpy()
            Q_values = Q_values.detach().cpu().numpy()
            for agent in range(len(Q_values)):
                probs = used_epsilon * numpy.ones(self.nr_actions) / self.nr_actions
                rest_prob = 1 - sum(probs)
                probs[argmax(Q_values[agent])] += rest_prob
                action_probs.append(probs / sum(probs))
            # print("11:{}".format(torch.cuda.memory_allocated(0)))
            return action_probs
        else:
            for i, agent_id in enumerate(agent_ids):
                history = [[joint_obs[i]] for joint_obs in histories]
                history = torch.tensor(history, device=self.device, dtype=torch.float32)
                # in_time = time.time()
                probs, value = self.policy_net(history)
                # print("1Once time is %s" % (time.time() - in_time))
                assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
                probs = probs.detach().cpu().numpy()[0]
                value = value.detach()
                action_probs.append(probs)
        return action_probs

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        returns = minibatch_data["pro_returns"]
        actions = minibatch_data["pro_actions"]
        old_probs = minibatch_data["pro_action_probs"]
        histories = minibatch_data["pro_histories"]
        action_probs, expected_values = self.policy_net(histories)
        policy_losses = []
        value_losses = []
        for probs, action, value, R, old_prob in zip(action_probs, actions, expected_values, returns, old_probs):
            value_index = 0
            if self.use_q_values:
                value_index = action
                advantage = value[value_index].detach()
            else:
                advantage = R - value.item()
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[value_index], torch.tensor(R)))
        value_loss = torch.stack(value_losses).mean()
        policy_loss = torch.stack(policy_losses).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True

    def policy_loss(self, advantage, probs, action, old_prob):
        m1 = Categorical(probs)
        m2 = Categorical(old_prob)
        ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        clipped_ratio = torch.clamp(ratio, 1-self.eps_clipping, 1+self.eps_clipping)
        surrogate_loss1 = ratio*advantage
        surrogate_loss2 = clipped_ratio*advantage
        return -torch.min(surrogate_loss1, surrogate_loss2)

    def value_update(self, minibatch_data):
        pass

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones):
        super(PPOLearner, self).update(state, observations, joint_action, rewards, next_state, next_observations, dones)
        global_terminal_reached = not [d for i,d in enumerate(dones) if (not d)]
        if global_terminal_reached and self.memory.size() > self.nr_episodes:
            is_protagonist = True
            trainable_setting = is_protagonist
            if self.params["UPDeT"]:
                m_size = self.memory.size()
                trainable_setting = trainable_setting and (m_size + 1 > self.batch_size)
            if trainable_setting:
                # 默认 capacity为 20000
                batch_size = self.params["batch_size"] if self.params["batch_size"] < self.memory.capacity else self.memory.capacity
                # batch = self.memory.sample_batch(batch_size)
                if self.params["UPDeT"]:
                    batch = self.memory.sample_time_batch(self.batch_size,
                                                              max_time_length=self.params["max_history_length"])
                else:
                    batch = self.memory.sample_batch(self.batch_size)

                minibatch_data = self.collect_minibatch_data(batch, whole_batch=True)

                self.value_update(minibatch_data)
                for _ in range(self.nr_epochs):
                    optimizer = self.protagonist_optimizer
                    self.policy_update(minibatch_data, optimizer)
                # if self.params["UPDeT"]:
                #     self.memory.clear()
            if not self.params["UPDeT"]:
                self.memory.clear()
            return True
        return False

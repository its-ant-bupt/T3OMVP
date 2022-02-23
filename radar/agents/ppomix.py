import random
import numpy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from radar.utils import get_param_or_default
from radar.agents.ppo import PPOLearner

class PPOMIXLearner(PPOLearner):

    def __init__(self, params):
        self.global_input_shape = params["global_observation_shape"]
        super(PPOMIXLearner, self).__init__(params)
        self.central_q_learner = params["central_q_learner"]
        self.last_q_loss = 0

    def value_update(self, minibatch_data):
        batch_size = minibatch_data["states"].size(0)
        self.central_q_learner.zero_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        nr_agents = self.get_nr_protagonists()
        if self.params["UPDeT"]:
            print(minibatch_data["pro_returns"].shape)
            returns = minibatch_data["pro_returns"].view(self.params["batch_size"], -1, nr_agents)
            returns = returns[:, -1].view(-1, nr_agents)
        else:
            returns = minibatch_data["pro_returns"].view(-1, nr_agents)
        returns = returns.gather(1, self.central_q_learner.zero_actions).squeeze()
        returns /= self.nr_agents
        returns *= nr_agents
        assert returns.size(0) == batch_size
        for _ in range(self.nr_epochs):
            self.last_q_loss = self.central_q_learner.train_step_with(minibatch_data, returns, nr_agents)

    def policy_update(self, minibatch_data, optimizer, random_agent_indices=None):
        old_probs = minibatch_data["pro_action_probs"]
        histories = minibatch_data["pro_histories"]
        actions = minibatch_data["pro_actions"]
        returns = minibatch_data["pro_returns"]
        if self.params["UPDeT"]:
            histories = histories.squeeze(dim=2)
            self.train_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().unsqueeze(0).expand(
                self.params["batch_size"], self.params["nr_agents"], 1, -1).cpu().numpy()
            time_agent_out = []
            time_expected_values = []
            for t in range(self.params["max_history_length"]):
                agent_out, h, expected_values = self.policy_net(histories[:, t],
                                               torch.tensor(self.train_hidden_state).reshape(self.params["batch_size"],
                                                                                             -1, 1,
                                                                                             self.params["emb"]).to(
                                                   self.device))
                self.train_hidden_state = h.cpu().detach().numpy()
                # agent_out = agent_out.view(self.params["batch_size"], self.params["nr_agents"], -1)
                # expected_values = expected_values.view(self.params["batch_size"], self.params["nr_agents"], -1)
                time_agent_out.append(agent_out)
                time_expected_values.append(expected_values)
            time_agent_out = torch.stack(time_agent_out, dim=1)
            time_expected_values = torch.stack(time_expected_values, dim=1)
            time_agent_out = time_agent_out.view(-1, self.params["nr_actions"])
            time_expected_values = time_expected_values.view(-1, 1)

            self.train_hidden_state = self.policy_net.protagonist_net.fc_net.init_hidden().unsqueeze(0).expand(
                self.params["batch_size"], self.params["nr_agents"], 1, -1).cpu().numpy()
            time_expected_Q_values = []
            for t in range(self.params["max_history_length"]):
                agent_out, h = self.central_q_learner.policy_net(histories[:, t],
                                               torch.tensor(self.train_hidden_state).reshape(self.params["batch_size"],
                                                                                             -1, 1,
                                                                                             self.params["emb"]).to(
                                                   self.device))
                self.train_hidden_state = h.cpu().detach().numpy()
                time_expected_Q_values.append(agent_out)
            time_expected_Q_values = torch.stack(time_expected_Q_values, dim=1)
            time_expected_Q_values = time_expected_Q_values.view(-1, self.params["nr_actions"]).detach()
            actions = actions.view(-1).type(torch.long)
            old_probs = old_probs.view(-1, self.params["nr_actions"])
            returns = returns.view(-1)
            policy_losses = []
            value_losses = []
            for probs, action, value, Q_values, old_prob, R in \
                    zip(time_agent_out, actions, time_expected_values, time_expected_Q_values, old_probs, returns):
                baseline = sum(probs * Q_values)
                baseline = baseline.detach()
                advantage = R - baseline
                policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
                value_losses.append(F.mse_loss(value[0], Q_values[action]))
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            action_probs, expected_values = self.policy_net(histories)
            expected_Q_values = self.central_q_learner.policy_net(histories).detach()
            policy_losses = []
            value_losses = []
            for probs, action, value, Q_values, old_prob, R in\
                zip(action_probs, actions, expected_values, expected_Q_values, old_probs, returns):
                baseline = sum(probs*Q_values)
                baseline = baseline.detach()
                advantage = R - baseline
                policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
                value_losses.append(F.mse_loss(value[0], Q_values[action]))
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return True

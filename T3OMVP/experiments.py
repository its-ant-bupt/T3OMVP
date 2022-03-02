import T3OMVP.data as data
from os.path import join
import numpy
import random
import T3OMVP.algorithm as algorithm
import T3OMVP.controller_loader as controller_loader
from torch.utils.tensorboard import SummaryWriter
import time
import torch


def log(log_level, message_level, message):
    if message_level <= log_level:
        print(message)

def produce_choice_list():
    list = []
    for i in range(4):
        for j in range(10):
            list.append(i)
    return list

def run_episode(episode_id, controller, params, training_mode=True, log_level=0, reset_episode=True, prey_choice = 0):
    env = params["env"]
    path = params["directory"]
    save_summaries = params["save_summaries"]
    nr_agents = params["nr_agents"]
    if reset_episode:
        observations = env.reset()
    else:
        observations = env.joint_observation()
    state = env.global_state()
    done = False
    time_step = 0
    state_summaries = [env.state_summary()]
    protagonist_discounted_return = 0
    protagonist_undiscounted_return = 0
    nr_protagonists = 1.0*(nr_agents)
    param = prey_choice
    while not done:
        joint_action = controller.policy(observations, training_mode)
        next_observations, rewards, dones, info = env.step(joint_action, param)
        protagonist_reward = sum([r/nr_protagonists for i,r in enumerate(rewards)])
        protagonist_discounted_return += (params["gamma"]**time_step)*protagonist_reward
        protagonist_undiscounted_return += protagonist_reward
        next_state = env.global_state()
        done = not [d for i,d in enumerate(dones) if (not d)]
        state_summary = env.state_summary()
        policy_updated = False
        if training_mode:
            policy_updated = controller.update(\
                state, observations, joint_action, rewards,\
                next_state, next_observations, dones)
        state = next_state
        observations = next_observations
        state_summary["transition_info"] = info
        state_summaries.append(state_summary)
        time_step += 1
    log(log_level, 0, "{} episode {} finished:\n\tdiscounted return: {}\n\tundiscounted return: {}\n\t domain statistics:{}"
        .format(params["domain_name"], episode_id, env.discounted_return, env.undiscounted_return, env.domain_statistic()))
    if save_summaries and training_mode and params["save_test_summaries"]:
        summary_filename = "episode_{}.json".format(episode_id)
        data.save_json(join(path, summary_filename), state_summaries)
        del state_summaries
    return protagonist_discounted_return, protagonist_undiscounted_return, policy_updated


def run_test(env, nr_test_episodes, controller, params, log_level):
    nr_protagonists = params["nr_agents"]
    test_discounted_returns = []
    test_undiscounted_returns = []
    test_domain_statistics = []
    for episode_id in range(nr_test_episodes):
        time1 = time.time()

        prey_choice_list = produce_choice_list()
        random.shuffle(prey_choice_list)
        prey_choice = prey_choice_list.pop()

        run_episode(episode_id="Test-{}".format(episode_id), controller=controller, params=params, training_mode=False, log_level=log_level, prey_choice=prey_choice)
        test_discounted_returns.append(env.discounted_return)
        test_undiscounted_returns.append(env.undiscounted_return)
        test_domain_statistics.append(env.domain_statistic())
        print("Time %s" % (time.time() - time1))
    return numpy.mean(test_discounted_returns)/nr_protagonists,\
        numpy.mean(test_undiscounted_returns)/nr_protagonists,\
        numpy.mean(test_domain_statistics)/nr_protagonists


def run_default_test(env, nr_test_episodes, controller, params, log_level):
    return run_test(env, nr_test_episodes, controller, params, log_level)


def run(controller, nr_episodes, params, log_level=0):
    env = params["env"]
    path = params["directory"]
    nr_test_episodes = params["nr_test_episodes"]
    test_suite = run_default_test
    training_discounted_returns = []
    training_undiscounted_returns = []
    test_discounted_returns = []
    test_undiscounted_returns = []
    protagonist_discounted_returns = []
    protagonist_undiscounted_returns = []
    domain_statistic = []
    test_domain_statistics = []
    summary_write = SummaryWriter(path)
    test_discounted_return, test_undiscounted_return, test_domain_statistic = \
        test_suite(env, nr_test_episodes, controller, params, log_level)
    test_discounted_returns.append(test_discounted_return)
    test_undiscounted_returns.append(test_undiscounted_return)
    test_domain_statistics.append(test_domain_statistic)
    nr_epoch_updates = 0
    best_test = 0.0
    for episode_id in range(nr_episodes):
        print("1:{}".format(torch.cuda.memory_allocated(0)))
        protagonist_discounted_return, protagonist_undiscounted_return, policy_updated =\
            run_episode(episode_id=episode_id, controller=controller, params=params, training_mode=True, log_level=log_level)
        if policy_updated:
            print("=====> updated in episode",(episode_id+1), "<=====")
            test_discounted_return, test_undiscounted_return, test_domain_statistic = \
                test_suite(env, nr_test_episodes, controller, params, log_level)
            test_discounted_returns.append(test_discounted_return)
            test_undiscounted_returns.append(test_undiscounted_return)
            test_domain_statistics.append(test_domain_statistic)
            summary_write.add_scalar('test_discounted_returns', test_discounted_return, nr_epoch_updates)
            summary_write.add_scalar('test_undiscounted_returns', test_undiscounted_return, nr_epoch_updates)
            summary_write.add_scalar('test_domain_statistics', test_domain_statistic, nr_epoch_updates)
            if nr_epoch_updates % 20 == 0:
                if test_domain_statistic > best_test:
                    print("Save the best example, epoch is %s" % nr_epoch_updates)
                    best_test = test_domain_statistic
                    best_path = join(path, "best.pth")
                    controller.save_weights_to_path(best_path)
            nr_epoch_updates += 1
        protagonist_discounted_returns.append(protagonist_discounted_return)
        protagonist_undiscounted_returns.append(protagonist_undiscounted_return)
        training_discounted_returns.append(env.discounted_return)
        training_undiscounted_returns.append(env.undiscounted_return)
        domain_statistic.append(env.domain_statistic())
        summary_write.add_scalar('protagonist_discounted_returns', protagonist_discounted_return, episode_id)
        summary_write.add_scalar('protagonist_undiscounted_returns', protagonist_undiscounted_return, episode_id)
        summary_write.add_scalar('training_discounted_returns', env.discounted_return, episode_id)
        summary_write.add_scalar('training_undiscounted_returns', env.undiscounted_return, episode_id)
        summary_write.add_scalar('training_domain_statistic', env.domain_statistic(), episode_id)
    log(log_level, 0, "DONE")
    return_values = {
        "training_discounted_returns":training_discounted_returns,
        "training_undiscounted_returns":training_undiscounted_returns,
        "test_discounted_returns":test_discounted_returns,
        "test_undiscounted_returns":test_undiscounted_returns,
        "protagonist_discounted_returns":protagonist_discounted_returns,
        "protagonist_undiscounted_returns":protagonist_undiscounted_returns,
        "training_domain_statistic":domain_statistic,
        "test_domain_statistic":test_domain_statistics
    }
    data.save_json(join(path, "returns.json"), return_values)
    controller.save_weights(path)
    return return_values

def test(controller, nr_episodes, params, log_level=0):
    env = params["env"]
    path = params["directory"]
    nr_test_episodes = params["nr_test_episodes"]
    test_suite = run_default_test
    test_discounted_returns = []
    test_undiscounted_returns = []
    test_domain_statistics = []
    test_discounted_return, test_undiscounted_return, test_domain_statistic = \
        test_suite(env, nr_test_episodes, controller, params, log_level)
    test_discounted_returns.append(test_discounted_return)
    test_undiscounted_returns.append(test_undiscounted_return)
    test_domain_statistics.append(test_domain_statistic)
    for episode_id in range(99):
        test_discounted_return, test_undiscounted_return, test_domain_statistic = \
            test_suite(env, nr_test_episodes, controller, params, log_level)
        test_discounted_returns.append(test_discounted_return)
        test_undiscounted_returns.append(test_undiscounted_return)
        test_domain_statistics.append(test_domain_statistic)

    log(log_level, 0, "DONE")
    return_values = {
        "test_discounted_returns":test_discounted_returns,
        "test_undiscounted_returns":test_undiscounted_returns,
        "test_domain_statistic":test_domain_statistics
    }
    data.save_json(join(path, "returns.json"), return_values)
    controller.save_weights(path)
    return return_values

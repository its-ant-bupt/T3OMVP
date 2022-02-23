import radar.domain as domain
import radar.algorithm as algorithm
import radar.experiments as experiments
import radar.data as data
import radar.utils as utils
import sys
import os
from os.path import join
from settings import params, nr_steps
import argparse
import numpy


def parse_args():
    parser = argparse.ArgumentParser("Experiments for multi-vehicle pursuit environments")
    parser.add_argument("--history_length", type=int, default=1, help="The observation length")
    parser.add_argument("--conv3d", action="store_true", default=False, help="Use conv3d")
    parser.add_argument("--time_transformer", action="store_true", default=False, help="Use time transformer")
    parser.add_argument("--UPDeT", action="store_true", default=False, help="Use UPDeT")
    parser.add_argument("--team_transformer", action="store_true", default=False, help="Use team transformer")
    parser.add_argument("--both_transformer", action="store_true", default=False, help="Use both transformer")
    parser.add_argument("--exp_name", type=str, default="test", help="adition name of the experiment")  # 实验名
    parser.add_argument("--batch_size", type=int, default=32, help="The train batch size")
    parser.add_argument("--domain_name", type=str, default="VehiclePursuit-8", help="The domain of training")
    parser.add_argument("--alg_name", type=str, default="AC-QMIX", help="The algorithm name of training")
    parser.add_argument("--lof", type=int, default=0, help="The format of observation")
    parser.add_argument("--reload", action="store_true", default=False, help="reload the model")
    parser.add_argument("--reload_exp", type=str, default=None, help="The reload exp name")
    parser.add_argument("--decoupling", action="store_true", default=False, help="Use the policy decoupling")
    parser.add_argument("--test", action="store_true", default=False, help="Test")
    return parser.parse_args()


args = parse_args()

params["conv3d"] = args.conv3d
params["timeTransformer"] = args.time_transformer
params["UPDeT"] = args.UPDeT
params["decoupling"] = args.decoupling
params["max_history_length"] = args.history_length
params["batch_size"] = args.batch_size
params["domain_name"] = args.domain_name
params["reload"] = args.reload
params["reload_exp"] = args.reload_exp
assert params["domain_name"] is not None, "domain_name is required"
params["algorithm_name"] = args.alg_name
params["local_observation_format"] = args.lof
params["nr_test_episodes"] = 50
params["nr_agents"] = 10
params["test"] = args.test

env = domain.make(params["domain_name"], params)
nr_episodes = int(nr_steps/env.time_limit)  # 40000
addInformation = ""
if params["timeTransformer"]:
    addInformation = "timeTransformer"

addInformation += "history-%s" % params["max_history_length"]
addInformation += "batch_size-%s" % params["batch_size"]

params["directory"] = "output/{}-agents_domain-{}_{}_{}".format(params["nr_agents"], params["domain_name"], params["algorithm_name"], addInformation)

params["directory"] = data.mkdir_with_expname(params["directory"], args.exp_name)
params["global_observation_shape"] = env.global_observation_space.shape
params["local_observation_shape"] = env.local_observation_space.shape
params["token_dim"] = numpy.prod(env.local_observation_space.shape)
if params["decoupling"]:
    params["token_dim"] = numpy.prod(env.local_observation_space.shape[1:])
params["nr_actions"] = env.action_space.n
params["gamma"] = env.gamma
params["env"] = env
controller = algorithm.make(params["algorithm_name"], params)
if params["reload"]:
    params["reload_exp"] = join("output", params["reload_exp"], "best.pth")
    if os.path.exists(params["reload_exp"]):
        controller.load_weights_from_history(params["reload_exp"])
        print("Load model success")
    else:
        print("Not found model")
if args.test:
    result = experiments.test(controller, nr_episodes, params, log_level=0)
else:
    result = experiments.run(controller, nr_episodes, params, log_level=0)

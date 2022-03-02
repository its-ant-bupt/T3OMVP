import T3OMVP.agents.controller as controller
import T3OMVP.agents.dqn as dqn
import T3OMVP.agents.vdn as vdn
import T3OMVP.agents.qmix as qmix

import argparse


def make(algorithm, params={}):
    if algorithm == "Random":
        params["adversary_ratio"] = 0
        return controller.Controller(params)
    if algorithm == "DQN":
        params["adversary_ratio"] = 0
        return dqn.DQNLearner(params)
    if algorithm == "VDN":
        params["adversary_ratio"] = 0
        return vdn.VDNLearner(params)
    if algorithm == "QMIX":
        params["adversary_ratio"] = 0
        return qmix.QMIXLearner(params)

    raise ValueError("Unknown algorithm '{}'".format(algorithm))
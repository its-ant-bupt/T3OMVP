import radar.environments.predator_prey as predator_prey
import radar.environments

def make(domain, params={}):
    if domain.startswith("PredatorPrey-"):
        return predator_prey.make(domain, params)
    elif domain.startswith("VehiclePursuit-"):
        return predator_prey.make(domain, params)
    raise ValueError("Environment '{}' unknown or not published yet".format(domain))
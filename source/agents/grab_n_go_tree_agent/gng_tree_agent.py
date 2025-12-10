from source.agents.tree.tree import TreeAgent
from source.agents.individual import *

import pickle

import os

# one could think of modifying all the hyperparameters here

class GNGTreeAgent(Individual):

    def __init__(self, catcher : TreeAgent, runner : TreeAgent, init_elo=100):
        super().__init__(init_elo)
        self.catcher = catcher
        self.runner = runner

    def need_map(self):
        return True

    def move(self, obs, eval_mode : bool = False, **kwargs):
        catcher = kwargs["catcher"]
        if catcher:
            return self.catcher.move(obs, eval_mode)
        else:
            return self.runner.move(obs, eval_mode)

    def save(self, path):
        d = os.path.dirname(path) 
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def mutate(self, prob_subtree=0.6, prob_node=0.3, prob_const=0.1):
        self.id = Individual._ids.__next__()
        self.catcher.mutate(prob_subtree=prob_subtree, prob_node=prob_node, prob_const=prob_const)
        self.runner.mutate(prob_subtree=prob_subtree, prob_node=prob_node, prob_const=prob_const)

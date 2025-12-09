"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    individual.py:
    
    This file contains the class for the individual representation.
    The mother class is the "Individual".
    Individuals have natively the attribute elo, since these individuals all will be inserted in an evolutionary ranking system.

    Classes list:
        - Individual
        - RealIndividual
        - RandomIndividual
        - GeneticPolicyIndividual
"""

from itertools import count
from gymnasium import Env
import copy
import pygame

class Individual():
    _ids = count(0)

    def __init__(self, init_elo = 100):
        self.init_elo = init_elo
        self.elo = init_elo
        self.id =  next(Individual._ids)
        self.n_matches = 0
        """self.elo_history = []
        self.opponent_history = []"""

    def get_elo(self):
        return self.elo
    
    def get_id(self):
        return self.id
    
    def update_elo(self, opponent_id, new_elo):
        """self.opponent_history.append(opponent_id)
        self.elo_history.append(self.elo)"""
        self.elo = max(new_elo, 0)
        self.n_matches += 1
    
    def reset_elo(self):
        self.elo = self.init_elo

    def overwrite(self, other):
        self.__dict__ = copy.deepcopy(other.__dict__)

    def observe(self, obs, action, rew, new_obs, done):
        pass

    def move(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def update(self, **kwrags):
        pass

    def save(self):
        pass

    @classmethod
    def load(cls):
        pass

import numpy as np
import copy
from itertools import count

class LogicalAIIndividual:
    _ids = count(0)

    def __init__(self, init_elo=100, lev = 1):
        self.elo = init_elo
        self.id = next(LogicalAIIndividual._ids)
        self.lev = lev
        self.n_matches = 0

        # Tuning thresholds (you can tweak them after testing)
        self.short_range = 10
        self.mid_range   = 30
        self.dodge_distance = 50

        # randomness
        self.random_move_prob  = 0.10
        self.random_punch_prob = 0.05

    def get_elo(self): return self.elo
    def get_id(self): return self.id
    
    def update_elo(self, opponent_id, new_elo):
        self.elo = max(new_elo, 0)
        self.n_matches += 1

    def overwrite(self, other):
        self.__dict__ = copy.deepcopy(other.__dict__)

    def observe(self, obs, action, rew, new_obs, done, **kwargs): pass
    def update(self, **kwargs): pass
    def save(self, **kwargs): pass

    # -----------------------------------------------------------------
    # MOVE FUNCTION WITH SHORT / MID / LONG PUNCH LOGIC
    # -----------------------------------------------------------------
    def move(self, env, perspective, **kwargs):
        """
            This is a simple yet effective bot. It has some level of difficulty tunable from 1 to 3.
        """
        
        self_xc, self_yc, opp_xc, opp_yc, opp_state = env._get_logical_info(perspective)

        if self.lev == 1:
            return np.random.randint(0, 8)
        
        threshold = 0.05

        if self.lev == 2:
            threshold = 0.5

        # -----------------------
        # RANDOM MOVEMENT
        # -----------------------
        if np.random.rand() < threshold:
            return np.random.randint(0, 5)  # random movement (0–4)

        # -----------------------
        # COMPUTE DISTANCES
        # -----------------------
        dx = opp_xc - self_xc
        dy = opp_yc - self_yc
        dist = np.sqrt(dx*dx + dy*dy)

        # -----------------------
        # DODGE OPPONENT PUNCH
        # -----------------------
        opponent_is_punching = opp_state in [5, 6, 7]

        if opponent_is_punching and dist < self.dodge_distance:
            # dodge vertically to avoid being predictable
            return 1 if np.random.rand() < 0.5 else 2

        # -----------------------
        # SELECT PUNCH TYPE
        # -----------------------
        if dist < self.short_range:
            # Short punch (closest)
            if np.random.rand() < self.random_punch_prob:
                return np.random.randint(5, 8)
            return 5

        if dist < self.mid_range:
            # Mid-range punch
            if np.random.rand() < self.random_punch_prob:
                return np.random.randint(5, 8)
            return 6

        if dist < self.mid_range * 1.5:
            # Long-range punch (if slightly further)
            if np.random.rand() < self.random_punch_prob:
                return np.random.randint(5, 8)
            return 7

        # -----------------------
        # MOVE TOWARD OPPONENT
        # -----------------------
        if abs(dx) > abs(dy):
            return 3 if dx < 0 else 4  # left / right
        else:
            return 1 if dy < 0 else 2  # up / down

        # -----------------------
        # DEFAULT (should not happen)
        # -----------------------S
        return 0

class RealIndividual(Individual):

    def move(self, env, **kwargs):
        """
        This move function will use the env.keymap attribute.
        the keymap is a dict mapping pygame keys to actions.
        """
        keys = pygame.key.get_pressed()

        # default no-op
        action = 0

        for key, action_value in env.keymap.items():
            if keys[key]:
                return action_value
        
        return action

class RandomIndividual(Individual):
    
    def move(self, game : Env, **kwargs):
        return game.action_space.sample()

if __name__ == '__main__':
    
    pass
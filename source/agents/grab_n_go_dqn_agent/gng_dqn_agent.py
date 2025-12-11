"""
    Developer : Bredariol Francesco

    gng_dqn_agent.py

    This file contains the implementation of an dqn agent for the grab n go environemnt. It is a wrapper of 2 differents dqn agents.
"""

from source.agents.dqn_agent.dqn_agent import DQNAgent
from source.agents.individual import *

import torch

import os

# one could think of modifying all the hyperparameters here

class GNGDQNAgent(Individual):

    def __init__(self, catcher : DQNAgent, runner : DQNAgent, init_elo=100):
        super().__init__(init_elo)
        self.catcher = catcher
        self.runner = runner

    def reset(self, percentage : float | None = None, **kwargs):
        if kwargs == {}:
            self.catcher.reset(percentage)
            self.runner.reset(percentage)
        else:
            catcher = kwargs["catcher"]
            if catcher:
                self.catcher.reset(percentage)
            else:
                self.runner.reset(percentage)

    def move(self, state, env, eval_mode : bool = False, **kwargs):
        catcher = kwargs["catcher"]
        if catcher:
            return self.catcher.move(state, env, eval_mode)
        else:
            return self.runner.move(state, env, eval_mode)
        
    def observe(self, obs, action, reward, next_obs, done, **kwargs):
        catcher = kwargs["catcher"]
        if catcher:
            self.catcher.observe(obs, action, reward, next_obs, done)
        else:
            self.runner.observe(obs, action, reward, next_obs, done)

    def update(self, **kwargs):
        catcher = kwargs["catcher"]
        if catcher:
            self.catcher.update()
        else:
            self.runner.update()
    
    def save(self, path):
        checkpoint = {
            "init_elo" : self.elo, 
            "catcher": {
                'elo': self.elo,
                'n_actions': self.catcher.n_actions,
                'n_observations': self.catcher.n_observations,
                'policy_state_dict': self.catcher.policy_net.state_dict(),
                'target_state_dict': self.catcher.target_net.state_dict(),
                'optimizer_state_dict': self.catcher.optimizer.state_dict(),
                'steps_done': self.catcher.steps_done,
            },
            "runner": {
                'elo': self.elo,
                'n_actions': self.runner.n_actions,
                'n_observations': self.runner.n_observations,
                'policy_state_dict': self.runner.policy_net.state_dict(),
                'target_state_dict': self.runner.target_net.state_dict(),
                'optimizer_state_dict': self.runner.optimizer.state_dict(),
                'steps_done': self.runner.steps_done,
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        data = torch.load(path, map_location=device, weights_only = False)

        elo = data["init_elo"]

        # reconstruct both DQNAgents from their sub-checkpoints
        catcher_data  = data["catcher"]
        runner_data = data["runner"]

        catcher = DQNAgent(
            n_actions=catcher_data['n_actions'],
            n_observations=catcher_data['n_observations'],
            init_elo=catcher_data['elo'],
            device=device
        )
        catcher.policy_net.load_state_dict(catcher_data['policy_state_dict'])
        catcher.target_net.load_state_dict(catcher_data['target_state_dict'])
        catcher.optimizer.load_state_dict(catcher_data['optimizer_state_dict'])
        catcher.steps_done = catcher_data['steps_done']

        runner = DQNAgent(
            n_actions=runner_data['n_actions'],
            n_observations=runner_data['n_observations'],
            init_elo=runner_data['elo'],
            device=device
        )
        runner.policy_net.load_state_dict(runner_data['policy_state_dict'])
        runner.target_net.load_state_dict(runner_data['target_state_dict'])
        runner.optimizer.load_state_dict(runner_data['optimizer_state_dict'])
        runner.steps_done = runner_data['steps_done']

        return cls(init_elo=elo, catcher=catcher, runner=runner)

    def mutate(self, policy_scale : float = 0.1, target_scale : float = 0.1):
        self.catcher.mutate(policy_scale=policy_scale, target_scale=target_scale)
        self.runner.mutate(policy_scale=policy_scale, target_scale=target_scale)

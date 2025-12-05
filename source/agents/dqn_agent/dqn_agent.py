"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    dqn_agent.py

    This file contains the implementation of an dqn agent.
"""
from source.agents.individual import Individual

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from source.agents.dqn_agent.replay_buffer import ReplayMemory, Transition

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 60*120*600 # fps * maximum time * minimum number of game to learn
TAU = 0.005
LR = 3e-4
REPLAY_SIZE = 60*60 # first 60 seconds of a game

class DQNAgent(Individual):

    def __init__(self, n_actions, n_observations, device = 'cuda' if torch.cuda.is_available() else 'cpu', init_elo=100):
        super().__init__(init_elo)

        # Get number of actions from gym action space
        self.n_actions = n_actions
        # Get the number of state observations
        self.n_observations = n_observations

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_SIZE)

        self.steps_done = 0

        self.device = device

        self.update_t = 0

    def reset(self, percentage = None):
        if percentage == None:
            self.steps_done -= max(0, 60*120*50) # just to reintroduce a bit of stochasticity
        else:
            self.steps_done = int(EPS_DECAY*percentage)
        self.update_t = 0

    def move(self, state, env, eval_mode = False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        sample = random.random()
        eps_threshold = EPS_START - (EPS_START - EPS_END) * (self.steps_done / EPS_DECAY)
        eps_threshold = max(EPS_END, eps_threshold)  # clamp so it doesn't go below end
        self.steps_done = min(self.steps_done + 1, EPS_DECAY) if not eval_mode else self.steps_done # to prevent overflow
        if sample > eps_threshold:
            with torch.no_grad():
                # torch.argmax().item() will return the index of the maximum
                act = torch.argmax(self.policy_net(state)).item()
                return act # torch.argmax(self.policy_net(state)).item()
        else:
            return env.action_space.sample()
    
    def observe(self, obs, action, reward, next_obs, done):

        self.memory.push(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor([action], dtype=torch.long),
        torch.tensor([reward], dtype=torch.float32),
        torch.tensor(next_obs, dtype=torch.float32),
        done
    )

    def update(self):

        self.update_t += 1

        if self.update_t%BATCH_SIZE != 0: # update only once every batch size steps
            return

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Stack batches
        state_batch = torch.stack(batch.state).to(self.device)              # [B, obs]
        next_state_batch = torch.stack(batch.next_state).to(self.device)         # [B, obs]
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)  # [B, 1]
        reward_batch = torch.stack(batch.reward).float().to(self.device)           # [B]
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)              # [B]

        # ------------------------------
        # Compute current Q(s, a)
        # ------------------------------
        q_values = self.policy_net(state_batch)                             # [B, num_actions]
        state_action_values = q_values.gather(1, action_batch).squeeze(1)   # [B]

        # ------------------------------
        # Compute target Q-values
        # ------------------------------
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1)[0]  # [B]
            target_q_values = reward_batch.squeeze(1) + GAMMA * next_q_values * (1 - done_batch) # [B]

        # ------------------------------
        # Compute loss
        # ------------------------------
        loss = F.smooth_l1_loss(state_action_values, target_q_values)

        # ------------------------------
        # Optimize the model
        # ------------------------------
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping (recommended for DQN stability)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

        self.optimizer.step()

        return loss.item()
    
    def save(self, path):
        """Save the agent to a file."""
        checkpoint = {
            'elo' : self.elo,
            'n_actions': self.n_actions,
            'n_observations': self.n_observations,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load an agent from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only = False)

        agent = cls(
            n_actions=checkpoint['n_actions'],
            n_observations=checkpoint['n_observations'],
            init_elo=checkpoint['elo'],
            device=device
        )

        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.steps_done = checkpoint['steps_done']

        return agent

    def mutate(self, policy_scale = 0.1, target_scale = 0.1):
        """
            This method should be used only by loaded individuals.
        """
        def perturb_model(model, scale=0.01):
            with torch.no_grad():            # avoid tracking in autograd
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * scale)
        perturb_model(self.policy_net, scale=policy_scale)
        perturb_model(self.target_net, scale=target_scale)
        self.reset_elo()
        self.reset(percentage=0.75)

"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    boxing.py

    This file contain the implementation of the game "grab n go".
    The boxing environment follows the gymnasium protocol.

    Class list:
        - Grab N Go
"""
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame
from source.games.grab_n_go.players import Player

FPS = 60
FRAME_DELAY = 1 # frame to wait between each decision -> this must be set also in the individual 
MAXIMUM_TIME = 30 # time in second
N_OBSTACLES = 5
W = 500
H = 500
OBSTACLE_SIZE = 50
OBSTACLE_COLOR = (255, 165, 0)
# the total maximum time in terms of time step is maximum_time * fps // frame_delay 

# ===========================================================
# GrabNGo Environment
# ===========================================================

class GrabNGoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode = "human", width=W, height=H, n_obstacles=N_OBSTACLES, max_time=MAXIMUM_TIME):
        
        super().__init__()

        # ------------------------------------------------------
        # Screen size and board details
        # ------------------------------------------------------
        self.width = width
        self.height = height
        self.n_obstacles = n_obstacles
        self.time = 0
        self.max_time = max_time

        self.p1 = Player(self.width//2 - 50 - 50//2, self.height//2, "P1", "catcher") # - 50 - half its width
        self.p2 = Player(self.width//2 + 50 - 50//2, self.height//2, "P2", "runner")

        # ------------------------------------------------------
        # Gym spaces (two players, discrete actions)
        # 0 = idle, 1 up, 2 down, 3 left, 4 right,
        # ------------------------------------------------------

        self.action_space = spaces.Discrete(5)

        # ------------------------------------------------------
        # Keymap used by real users
        # ------------------------------------------------------

        self.keymap = {
            pygame.K_w : 1,
            pygame.K_s : 2, 
            pygame.K_a : 3, 
            pygame.K_d : 4
        }

        # For now the observation space is composed by :
        # p1.state, p2.state, time feature
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(Player.state_dim()*2 + 1 + 2*self.n_obstacles,),   # twice the size of a player + time
            dtype=np.float32
        )

        # ------------------------------------------------------
        # Rendering
        # ------------------------------------------------------
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.reset()

    def get_obs(self, perspective = None):
        """
            if no perspective is required, a standard representation is given. 
            p1.state, p2.state, time_feature, ring_corners
        """
        assert perspective == None or perspective == 'p1' or perspective == 'p2'

        TIME_CAP = MAXIMUM_TIME * FPS // FRAME_DELAY
        time = (2//TIME_CAP * self.time) - 1 # linear compression of the time
        
        if perspective == None:
            return np.array([*self.p1.get_state(), *self.p2.get_state(), *self.get_obstacles(), time])
        if perspective == 'p1':
            return np.array([*self.p1.get_state(), *self.p2.get_state(), *self.get_obstacles(), time])
        if perspective == 'p2':
            return np.array([*self.p2.get_state(), *self.p1.get_state(), *self.get_obstacles(), time])
        
    def _place_random_objects(self):
        
        self.obstacles = []
        for _ in range (self.n_obstacles):
            ox, oy = np.random.randint([0, 0], [self.width, self.height])
            obstacle = pygame.Rect(ox, oy, OBSTACLE_SIZE, OBSTACLE_SIZE)
            while obstacle.colliderect(self.p1.get_rect()) or obstacle.colliderect(self.p2.get_rect()):
                ox, oy = np.random.randint([0, 0], [self.width, self.height])
                obstacle = pygame.Rect(ox, oy, OBSTACLE_SIZE, OBSTACLE_SIZE)
            self.obstacles.append(obstacle)

    def get_obstacles(self):
        return [coord for obstacle in self.obstacles for coord in obstacle.center]
    
    # =======================================================
    # Reset
    # =======================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.p1 = Player(self.width//2 - 100 - 50//2, self.height//2, "P1", "catcher") # - 50 - half its width
        self.p2 = Player(self.width//2 + 100 - 50//2, self.height//2, "P2", "runner")

        self._place_random_objects()

        self.time = 0

        return self.get_obs(), {}

    def step(self, actions):

        if actions == None : #default
            a1, a2 = 0, 0
        else:
            a1, a2 = actions

        reward_p1, reward_p2 = 0, 0

        # -------------------------------
        # Movement
        # -------------------------------
        def legit_movement(p : Player, action):
            t = Player(p.x, p.y, p.name, p.role)
            t.move(action)
            for obstacle in self.obstacles:
                if t.get_rect().colliderect(obstacle):
                    return False
            return True
        
        if legit_movement(self.p1, a1): # This should be tracked in some way
            self.p1.move(a1)

        if legit_movement(self.p2, a2):
            self.p2.move(a2)

        # --------------------------------
        # Caught
        # --------------------------------

        caught = self.p1.get_rect().colliderect(self.p2.get_rect())

        self.time += 1

        terminated = False

        # ----------------------------------
        # Termination
        # ----------------------------------

        if caught:
            terminated = True
            if self.p1.role == 'catcher':
                reward_p1 = 1
                reward_p2 = -1
            else:
                reward_p1 = -1
                reward_p2 = 1
        elif self.time >= FPS*MAXIMUM_TIME//FRAME_DELAY:
            terminated = True
            if self.p2.role == 'catcher':
                reward_p1 = 1
                reward_p2 = -1
            else:
                reward_p1 = -1
                reward_p2 = 1

        return self.get_obs(), (reward_p1, reward_p2), terminated, False, {}

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.window, OBSTACLE_COLOR, obs)

        # Draw players
        pygame.draw.rect(
            self.window,
            self.p1.get_color(),
            self.p1.get_rect()
        )

        pygame.draw.rect(
            self.window,
            self.p2.get_color(),
            self.p2.get_rect()
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

if __name__ == '__main__':
    
    env = GrabNGoEnv()

    done = False

    env.render()

    while not done:
        a1 = env.action_space.sample()
        a2 = env.action_space.sample()
        _, _, done, _, _ = env.step((a1, a2))
        env.render()
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

from source.games import UserClosingWindowException

FPS = 60
FRAME_DELAY = 1 # frame to wait between each decision -> this must be set also in the individual 
MAXIMUM_TIME = 30 # time in second
N_OBSTACLES = 0
W = 500
H = 500
OBSTACLE_SIZE = 50
OBSTACLE_COLOR = (255, 165, 0)
PLAYERS_DISTANCE = 60
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
        self.grid = pygame.Rect(0, 0, width, height)
        self.n_obstacles = n_obstacles
        self.time = 0
        self.max_time = max_time

        self.p1 = Player(self.width//2 - PLAYERS_DISTANCE//2 - 50//2, self.height//2, "P1", "catcher") # - 50 - half its width
        self.p2 = Player(self.width//2 + PLAYERS_DISTANCE//2 - 50//2, self.height//2, "P2", "runner")

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

    def get_obs(self, perspective = None, map = False):
        """
            if no perspective is required, a standard representation is given. 
            p1.state, p2.state, time_feature, ring_corners
        """
        assert perspective == None or perspective == 'p1' or perspective == 'p2'

        TIME_CAP = MAXIMUM_TIME * FPS // FRAME_DELAY
        time = (2//TIME_CAP * self.time) # linear compression of the time

        if map:
            p1_x, p1_y = self.p1.get_state()
            p2_x, p2_y = self.p2.get_state()
            if perspective == None:
                return {'p1_x' : p1_x, 'p1_y' : p1_y, 'p2_x' : p2_x, 'p2_y' : p2_y, 'speed' : self.p1.speed, 'size' : self.p1.size, 'time' : FRAME_DELAY*self.time//FPS, 'maximum_time' : MAXIMUM_TIME, 'width' : self.width, 'height' : self.height}
            if perspective == 'p1':
                return {'x' : p1_x, 'y' : p1_y, 'dx' : p1_x - p2_x, 'dy' : p1_y - p2_y, 'remaining_x' : self.width - p1_x, 'remaining_y' : self.height - p1_y}
            if perspective == 'p2':
                return {'x' : p2_x, 'y' : p2_y, 'dx' : p2_x - p1_x, 'dy' : p2_y - p1_y, 'remaining_x' : self.width - p2_x, 'remaining_y' : self.height - p2_y}
        else:
            if perspective == None:
                return np.array([*self.p1.get_state(), *self.p2.get_state(), *self.get_obstacles(), time])
            if perspective == 'p1':
                return np.array([*self.p1.get_state(), *self.p2.get_state(), *self.get_obstacles(), time])
            if perspective == 'p2':
                return np.array([*self.p2.get_state(), *self.p1.get_state(), *self.get_obstacles(), time])
        
    def _not_ok(self, obstacle : pygame.Rect):
            
            p1_collision = obstacle.colliderect(self.p1.get_rect())
            p2_collision = obstacle.colliderect(self.p2.get_rect())
            inside = not self.grid.contains(obstacle)
            
            if len(self.obstacles) == 0:
                return p1_collision or p2_collision or inside
            
            
            others_collision = not obstacle.collidelist(self.obstacles) == -1

            return  p1_collision or p2_collision or others_collision or inside
        
    def _place_random_objects(self):
        
        self.obstacles = []

        for _ in range (self.n_obstacles):
            ox, oy = np.random.randint([0, 0], [self.width, self.height])
            obstacle = pygame.Rect(ox, oy, OBSTACLE_SIZE, OBSTACLE_SIZE)
            while self._not_ok(obstacle):
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

        self.p1 = Player(self.width//2 - PLAYERS_DISTANCE//2 - 50//2, self.height//2, "P1", "catcher") # - 50 - half its width
        self.p2 = Player(self.width//2 + PLAYERS_DISTANCE//2 - 50//2, self.height//2, "P2", "runner")

        self._place_random_objects()

        self.time = 0

        return self.get_obs(), {}

    def step(self, actions):

        def players_distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        old_dist = players_distance(self.p1, self.p2)

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
            return self.grid.contains(t.get_rect())
        
        info = {'a1' : a1, 'a2' : a2}
        
        if legit_movement(self.p1, a1): # This should be tracked in some way
            self.p1.move(a1)
        else:
            self.p1.move(0)
            """reward_p1 -= 1"""
            info['a1'] = 0

        if legit_movement(self.p2, a2):
            self.p2.move(a2)
        else:
            self.p2.move(0)
            """reward_p2 -= 1"""
            info['a2'] = 0

        # -------------------------------
        # Reward shaping
        # -------------------------------
        """new_dist = players_distance(self.p1, self.p2)

        reward_p1 += 0.1 if new_dist > old_dist and self.p1.role == 'runner' else -0.1
        reward_p2 += 0.1 if new_dist > old_dist and self.p2.role == 'runner' else -0.1"""

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
                self.p1.score = 1
                reward_p1 = 10
                reward_p2 = -10
            else:
                self.p2.score = 1
                reward_p1 = -10
                reward_p2 = 10
        elif self.time >= FPS*MAXIMUM_TIME//FRAME_DELAY:
            terminated = True
            if self.p2.role == 'catcher':
                self.p1.score = 1
                reward_p1 = 10
                reward_p2 = -10
            else:
                self.p2.score = 1
                reward_p1 = -10
                reward_p2 = 10

        return self.get_obs(), (reward_p1, reward_p2), terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return 
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise UserClosingWindowException("The game is terminated since the user decided to quit")

        self.window.fill((0, 0, 0))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.window, OBSTACLE_COLOR, obs)

        # Draw players
        font = pygame.font.Font(None, 32)
        pygame.draw.rect(
            self.window,
            self.p1.get_color(),
            self.p1.get_rect()
        )
        p1_surf = font.render(f"{self.p1.name}", True, (0, 0, 0))
        p1_rect = p1_surf.get_rect(center = self.p1.get_rect().center)
        self.window.blit(p1_surf, p1_rect)

        pygame.draw.rect(
            self.window,
            self.p2.get_color(),
            self.p2.get_rect()
        )
        p2_surf = font.render(f"{self.p2.name}", True, (0, 0, 0))
        p2_rect = p2_surf.get_rect(center = self.p2.get_rect().center)
        self.window.blit(p2_surf, p2_rect)

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
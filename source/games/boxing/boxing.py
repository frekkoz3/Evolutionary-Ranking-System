"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    boxing.py

    This file contain the implementation of the game "boxing".

    Class list:
        - BoxingEnv
"""
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from .boxers import Boxer

class UserClosingWindowException(Exception):
    pass

FPS = 60
FRAME_DELAY = 1 # frame to wait between each decision
MAXIMUM_TIME = 120 # time in second
# the total maximum time in terms of time step is maximum_time * fps // frame_delay 

# ===========================================================
# Boxing Environment
# ===========================================================
class BoxingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()

        # ------------------------------------------------------
        # Gym spaces (two players, discrete actions)
        # 0 = idle, 1 up, 2 down, 3 left, 4 right,
        # 5 = jab, 6 = hook, 7 = uppercut
        # ------------------------------------------------------
        self.action_space = spaces.Discrete(8)

        # Keymap used by real users
        self.keymap = {
            pygame.K_w : 1,
            pygame.K_s : 2, 
            pygame.K_a : 3, 
            pygame.K_d : 4,
            pygame.K_q : 5,
            pygame.K_e : 6,
            pygame.K_r : 7 
        }

        # Observation could later be replaced by game state arrays
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(300, 600, 3),
            dtype=np.uint8
        )

        # ------------------------------------------------------
        # Rendering
        # ------------------------------------------------------
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # screen sizes
        self.W = 450
        self.H = 500

        # Ring bounds:
        # leave a HUD zone of 50px at top for points
        self.HUD_HEIGHT = 50
        self.RING_LEFT = 20
        self.RING_TOP = 50
        self.RING_RIGHT = self.W - 20
        self.RING_BOTTOM = self.H - 20

        # Timer
        self.time = 0

        # Initialize boxers
        self.reset()

    # =======================================================
    # Reset
    # =======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.p1 = Boxer(self.W//2 - 50 - 50//2, self.H//2 - self.HUD_HEIGHT, (200, 50, 50), "P1") # - 50 - half its width
        self.p2 = Boxer(self.W//2 + 50 - 50//2, self.H//2 - self.HUD_HEIGHT, (50, 50, 200), "P2")

        self.time = 0

        obs = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        return obs, {}

    # =======================================================
    # Step
    # =======================================================
    def step(self, actions):

        if actions == None : #default
            a1, a2 = 0, 0
        else:
            a1, a2 = actions

        # -------------------------------
        # Movement
        # -------------------------------

        def legit_movement(p1 : Boxer, p2 : Boxer, action):
            """
                This function simulate the movement one step ahead in the future. 
                If a player collaps over an other player it return False, meaning the action is invalid.
            """
            pt = Boxer(p1.x, p1.y, p1.color, name="temp")
            pt.move(action)
            return not pt.get_rect().colliderect(p2.get_rect())
        
        if legit_movement(self.p1, self.p2, a1):
            self.p1.move(a1)
        if legit_movement(self.p2, self.p1, a2):
            self.p2.move(a2)

        self._clamp_in_ring(self.p1)
        self._clamp_in_ring(self.p2)

        # -------------------------------
        # Punch attempts
        # -------------------------------
        if a1 in [5, 6, 7]:
            self.p1.start_punch(a1 - 5)

        if a2 in [5, 6, 7]:
            self.p2.start_punch(a2 - 5)

        # -------------------------------
        # Punch update
        # -------------------------------
        self.p1.update_punch_state(facing_left=False)
        self.p2.update_punch_state(facing_left=True)

        # -------------------------------
        # Hit detection
        # -------------------------------

        def hit_detection(p1 : Boxer, p2 : Boxer):
            if p1.hitbox and p1.hitbox.colliderect(p2.get_rect()):
                if p2.state == 1: # combo reset
                    p2.cancel_punch()
                p1.score += 1
                return +1, -0.1 # penalizing to get hit
            return 0, 0
        
        reward_p1, reward_p2 = hit_detection(self.p1, self.p2)
        t1, t2 = hit_detection(self.p2, self.p1)
        reward_p1 += t1
        reward_p2 += t2

        # -------------------------------
        # Regenerate stamina slowly
        # -------------------------------
        self.p1.regenerate()
        self.p2.regenerate()

        # -------------------------------
        # Check ending
        # -------------------------------
        terminated = False # to add the truncation for the time
        if self.p1.score >= 100 or self.p2.score >= 100:
            terminated = True

        obs = np.zeros((self.H, self.W, 3), dtype=np.uint8) # this must be modified!!
        reward = (reward_p1, reward_p2)

        # --------------------------------
        # Time update
        # --------------------------------
        self.time += 1
        if self.time >= MAXIMUM_TIME * FPS // FRAME_DELAY:
            return obs, reward, True, True, {} 
        
        return obs, reward, terminated, False, {}

    # =======================================================
    # Clamp player in the ring area
    # =======================================================
    def _clamp_in_ring(self, p):
        p.x = max(self.RING_LEFT, min(self.RING_RIGHT - p.size, p.x))
        p.y = max(self.RING_TOP, min(self.RING_BOTTOM - p.size, p.y))

    # =======================================================
    # Render
    # =======================================================
    def render(self):
        if self.render_mode != "human":
            return 
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("BoxingEnv")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise UserClosingWindowException("The game is terminated since the user decided to quit")

        self.window.fill((30, 30, 30))

        # --------------------------------------------------
        # Draw ring (simple rectangle)
        # --------------------------------------------------
        pygame.draw.rect(
            self.window,
            (80, 80, 80),
            pygame.Rect(
                self.RING_LEFT,
                self.RING_TOP,
                (self.RING_RIGHT - self.RING_LEFT),
                (self.RING_BOTTOM - self.RING_TOP)
            ),
            3,
        )

        # --------------------------------------------------
        # Draw players (sprite if exists, else rectangle)
        # --------------------------------------------------
        for boxer in [self.p1, self.p2]:
            if boxer.sprite is None:
                pygame.draw.rect(self.window, boxer.color, boxer.get_rect())
            else:
                self.window.blit(boxer.sprite, (boxer.x, boxer.y))

            # Draw hitbox
            if boxer.hitbox:
                pygame.draw.rect(self.window, (255, 0, 0), boxer.hitbox, 2)

        # --------------------------------------------------
        # Draw score / HUD
        # --------------------------------------------------
        font = pygame.font.SysFont("Arial", 28) #Font("./fonts/PressStart2P.ttf", 28) # to solve the path issues
        text_color = (255, 255, 255)
        p1_score = font.render(f"{self.p1.score}", True, text_color)
        p2_score = font.render(f"{self.p2.score}", True, text_color)
        divider = font.render("|", True, text_color)
        self.window.blit(p1_score, (self.W//2 - 65, 10))
        self.window.blit(divider, (self.W//2 - 15, 10))
        self.window.blit(p2_score, (self.W//2 + 35, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    # =======================================================
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

if __name__ == '__main__':
    
    env = BoxingEnv(render_mode="human")
    obs, info = env.reset()

    done = False

    t = 0

    while not done:    
        
        actions = (env.action_space.sample(), env.action_space.sample())

        obs, rewards, done, truncated, info = env.step((4, 3))

        try:
            env.render()
        except UserClosingWindowException as e:
            done, truncated = True, True
        t+=1
        if t > 180:
            done, truncated = True, True

    env.close()

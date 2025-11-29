"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    boxing.py

    This file contain the implementation of the game "boxing".
    The boxing environment follows the gymnasium protocol.

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
FRAME_DELAY = 1 # frame to wait between each decision -> this must be set also in the individual 
MAXIMUM_TIME = 120 # time in second
# the total maximum time in terms of time step is maximum_time * fps // frame_delay 

# ===========================================================
# Boxing Environment
# ===========================================================
class BoxingEnv(gym.Env):

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()

        # ------------------------------------------------------
        # Gym spaces (two players, discrete actions)
        # 0 = idle, 1 up, 2 down, 3 left, 4 right,
        # 5 = short punch, 6 = mid punch, 7 = long punch
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

        # For now the observation space is composed by :
        #  p1.state, p2.state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(Boxer.state_dim()*2,),   # twice the size of a player
            dtype=np.float32
        )
        # ------------------------------------------------------
        # Rendering
        # ------------------------------------------------------
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # screen sizes
        self.W = 300
        self.H = 350

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

    def get_obs(self, perspective = None):
        """
            if no perspective is required, a standard representation is given. 
            p1.state, p2.state 
        """
        assert perspective == None or perspective == 'p1' or perspective == 'p2'
         
        if perspective == None:
            return np.array([*self.p1.get_state(), *self.p2.get_state()])
        if perspective == 'p1':
            return np.array([*self.p1.get_state(), *self.p2.get_state()])
        if perspective == 'p2':
            return np.array([*self.p2.get_state(), *self.p1.get_state()])
        
    def _get_logical_info(self, perspective):
        """
            This can be used only by the LogicalAIBot for training purpose only.
        """
        if perspective == 'p1':
            return *self.p1.get_rect().center, *self.p2.get_rect().center, self.p2.state
        else:
            return *self.p2.get_rect().center, *self.p1.get_rect().center, self.p1.state

    # =======================================================
    # Reset
    # =======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.p1 = Boxer(self.W//2 - 50 - 50//2, self.H//2 - self.HUD_HEIGHT, (200, 50, 50), "P1") # - 50 - half its width
        self.p2 = Boxer(self.W//2 + 50 - 50//2, self.H//2 - self.HUD_HEIGHT, (50, 50, 200), "P2")

        self.time = 0

        return self.get_obs(), {}

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
        
        movement_learning_1 = -1
        movement_learning_2 = -1
        
        if legit_movement(self.p1, self.p2, a1):
            self.p1.move(a1)
            movement_learning_1 = 0
        if legit_movement(self.p2, self.p1, a2):
            self.p2.move(a2)
            movement_learning_2 = 0

        self._clamp_in_ring(self.p1)
        self._clamp_in_ring(self.p2)

        # -------------------------------
        # Punch attempts
        # -------------------------------
        stamina_penalty_1 = 0
        stamina_penalty_2 = 0

        if a1 in [5, 6, 7]:
            if self.p1.start_punch(a1 - 5) == -1:
                stamina_penalty_1 = -1

        if a2 in [5, 6, 7]:
            if self.p2.start_punch(a2 - 5) == -1:
                stamina_penalty_2 = -1

        # -------------------------------
        # Punch update
        # -------------------------------
        if self.p1.get_rect().centerx > self.p2.get_rect().centerx:
            self.p1.update_punch_state(facing_left=True)
            self.p2.update_punch_state(facing_left=False)
        else:
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
                return +10, -1 # penalizing to get hit
            if p1.hitbox and not p1.hitbox.colliderect(p2.get_rect()):
                return -1, 0 # penalizing missed punches  
            return 0, 0
        
        reward_p1, reward_p2 = hit_detection(self.p1, self.p2)
        t1, t2 = hit_detection(self.p2, self.p1)
        reward_p1 += t1 + movement_learning_1 + stamina_penalty_1
        reward_p2 += t2 + movement_learning_2 + stamina_penalty_2

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
            if self.p1.score > self.p2.score:
                reward_p1 = 100
                reward_p2 = -100
            elif self.p2.score > self.p1.score:
                reward_p1 = -100
                reward_p2 = 100
            else:
                reward_p1 = -1
                reward_p2 = -1
        
        reward = (reward_p1, reward_p2)

        # --------------------------------
        # Time update
        # --------------------------------
        self.time += 1
        if self.time >= MAXIMUM_TIME * FPS // FRAME_DELAY:
            if self.p1.score > self.p2.score:
                reward_p1 = 100
                reward_p2 = -100
            elif self.p2.score > self.p1.score:
                reward_p1 = -100
                reward_p2 = 100
            else:
                reward_p1 = -1
                reward_p2 = -1
            reward = (reward_p1, reward_p2)
            return self.get_obs(), reward, True, True, {} 
        
        return self.get_obs(), reward, terminated, False, {}

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
        # Draw score
        # --------------------------------------------------
        font = pygame.font.SysFont("Arial", 28) #Font("./fonts/PressStart2P.ttf", 28) # to solve the path issues
        text_color = (255, 255, 255)
        p1_score = font.render(f"{self.p1.score}", True, text_color)
        p2_score = font.render(f"{self.p2.score}", True, text_color)
        divider = font.render("|", True, text_color)
        self.window.blit(p1_score, (self.W//2 - 65, 10))
        self.window.blit(divider, (self.W//2 - 15, 10))
        self.window.blit(p2_score, (self.W//2 + 35, 10))

        # ----------------------------------------------------
        # Draw Stamina
        # ----------------------------------------------------
        def color_by_stamina(stamina):
            if stamina > 50:
                return (0, 255, 0)
            elif stamina > 25:
                return (255, 255, 0)
            else:
                return (255, 0, 0)
        text_color = (255, 255, 0)
        p1_score = font.render(f"{int(self.p1.stamina)}", True, color_by_stamina(self.p1.stamina))
        p2_score = font.render(f"{int(self.p2.stamina)}", True, color_by_stamina(self.p2.stamina))
        self.window.blit(p1_score, (25, 10))
        self.window.blit(p2_score, (self.W - 60, 10))

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

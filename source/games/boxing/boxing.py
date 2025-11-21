
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


class BoxingEnv(gym.Env):
    """
    Two-player boxing environment following Gym protocol.
    step(a1, a2) => both players act
    step(a1)     => a2 generated automatically (simple AI)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Screen size
        self.W, self.H = 800, 700
        # Ring size
        self.HUD_HEIGHT = 80   # space on top for score
        self.RING_LEFT = 50
        self.RING_TOP = self.HUD_HEIGHT
        self.RING_WIDTH = self.W - 100
        self.RING_HEIGHT = self.H - self.HUD_HEIGHT - 50

        # Observation:
        # [p1_x, p1_y, p1_stamina, p1_state,
        #  p2_x, p2_y, p2_stamina, p2_state]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.W, self.H, 100, 3, self.W, self.H, 100, 3], dtype=np.float32),
            shape=(8,),
        )

        # Both players use the same action space
        # 0 = no-op
        # 1 = up
        # 2 = down
        # 3 = left
        # 4 = right
        # 5 = jab
        # 6 = hook
        # 7 = uppercut
        self.action_space = spaces.Discrete(8)

        # Internal initialization
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()

            import os

            # Load sprites
            """self.sprites = {}

            for player in ["p1", "p2"]:
                self.sprites[player] = {
                    "idle": pygame.image.load(os.path.join("sprites", f"{player}_idle.png")).convert_alpha(),
                    "jab": pygame.image.load(os.path.join("sprites", f"{player}_jab.png")).convert_alpha(),
                    "hook": pygame.image.load(os.path.join("sprites", f"{player}_hook.png")).convert_alpha(),
                    "uppercut": pygame.image.load(os.path.join("sprites", f"{player}_uppercut.png")).convert_alpha(),
                }"""
            
            self.screen = pygame.display.set_mode((self.W, self.H))
            self.clock = pygame.time.Clock()

        self.reset()

    # ---------------------------------------------------------
    # Boxer definition
    # ---------------------------------------------------------
    def make_boxer(self, x, y):
        return {
            "x": x,
            "y": y,
            "score" : 0,
            "size": 40,
            "speed": 5,
            "stamina": 100,
            "max_stamina": 100,
            "state": 0,    # 0 idle, 1 startup, 2 active, 3 recovery
            "timer": 0,
            "punch": None
        }

    # punch parameters
    PUNCHES = { # startup time, active time, recovery time, stamina cost
        0: (5, 4, 8, 5),    # jab
        1: (10, 4, 14, 12), # hook
        2: (14, 4, 18, 16)  # uppercut
    }

    def start_punch(self, boxer, punch_type):
        if boxer["state"] != 0: # punch can only start in startup state
            return

        startup, active, recovery, cost = self.PUNCHES[punch_type]

        if boxer["stamina"] < cost:
            return

        boxer["stamina"] -= cost
        boxer["state"] = 1   # startup
        boxer["timer"] = startup
        boxer["punch"] = punch_type

    def update_punch_state(self, boxer):
        if boxer["state"] == 0:
            return

        boxer["timer"] -= 1

        if boxer["state"] == 1 and boxer["timer"] <= 0:
            # startup → active
            _, active, _, _ = self.PUNCHES[boxer["punch"]]
            boxer["state"] = 2
            boxer["timer"] = active

        elif boxer["state"] == 2 and boxer["timer"] <= 0:
            # active → recovery
            _, _, recovery, _ = self.PUNCHES[boxer["punch"]]
            boxer["state"] = 3
            boxer["timer"] = recovery

        elif boxer["state"] == 3 and boxer["timer"] <= 0:
            boxer["state"] = 0
            boxer["punch"] = None

    # ---------------------------------------------------------
    # Gym required methods
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.p1 = self.make_boxer(
            self.RING_LEFT + 100,
            self.RING_TOP + self.RING_HEIGHT // 2,
        )

        self.p2 = self.make_boxer(
            self.RING_LEFT + self.RING_WIDTH - 100,
            self.RING_TOP + self.RING_HEIGHT // 2,
        )
        self.t = 0

        return self._get_obs(), {}

    def step(self, action_p1, action_p2=None):
        """
        For two players:
            step(a1, a2)

        For single agent training:
            step(a1)
            -> a2 is generated automatically
        """
        self.t += 1

        # -----------------------------------------------------
        # If only one action is provided, handle enemy AI
        # -----------------------------------------------------
        if action_p2 is None:
            # simple heuristic enemy
            dx = np.sign(self.p1["x"] - self.p2["x"])
            dy = np.sign(self.p1["y"] - self.p2["y"])

            r = np.random.rand()
            if r < 0.02:
                action_p2 = np.random.choice([5, 6, 7])  # random punch
            else:
                # move toward player
                if abs(dx) > abs(dy):
                    action_p2 = 3 if dx < 0 else 4
                else:
                    action_p2 = 1 if dy < 0 else 2

        # -----------------------------------------------------
        # Apply actions to both fighters
        # -----------------------------------------------------
        self._apply_action(self.p1, action_p1)
        self._apply_action(self.p2, action_p2)

        # Update punching states & stamina regen
        for b in (self.p1, self.p2):
            self.update_punch_state(b)
            b["stamina"] = min(b["max_stamina"], b["stamina"] + 0.2)

        # -----------------------------------------------------
        # Hit detection & reward
        # -----------------------------------------------------
        reward_p1, reward_p2 = 0, 0

        if self.p1["state"] == 2 and self._collide(self.p1, self.p2):
            if self.p2["state"] == 1:
                self.p2["state"] = 0 # punch cancel (TO CHECK IF IT WORKS)
                self.p2["punch"] = None
                reward_p1 += 0.5
            self.p1["score"] += 1
            self.p2["stamina"] -= 10
            reward_p1 += 1
            reward_p2 -= 1

        if self.p2["state"] == 2 and self._collide(self.p2, self.p1):
            if self.p1["state"] == 1:
                self.p1["state"] = 0 # punch cancel
                self.p1["punch"] = None
                reward_p2 += 0.5
            self.p2["score"] += 1
            self.p1["stamina"] -= 10
            reward_p2 += 1
            reward_p1 -= 1

        # -----------------------------------------------------
        # Terminal conditions
        # -----------------------------------------------------
        done = False # to add truncated 
        if self.p2["score"] > 100:
            reward_p2 += 10
            reward_p1 -= 10
            done = True

        if self.p1["score"] > 100:
            reward_p1 += 10
            reward_p2 -= 10
            done = True

        obs = self._get_obs()

        # Truncated always False for now
        return obs, (reward_p1, reward_p2), done, False, {}

    # ---------------------------------------------------------
    # Action helper
    # ---------------------------------------------------------
    def _apply_action(self, boxer, action):
        if boxer["stamina"] <= 0:
            boxer["state"] = 0
            boxer["punch"] = None
            action = 0
            boxer["stamina"] += 1
        # movement
        if action in [1, 2, 3, 4]:
            boxer["stamina"] -= 0.5 # little stamina cost for the movement
        if action == 1:  # up
            boxer["y"] -= boxer["speed"]
        elif action == 2:  # down
            boxer["y"] += boxer["speed"]
        elif action == 3:  # left
            boxer["x"] -= boxer["speed"]
        elif action == 4:  # right
            boxer["x"] += boxer["speed"]

        # boundary clamp
        boxer["x"] = np.clip(
            boxer["x"],
            self.RING_LEFT,
            self.RING_LEFT + self.RING_WIDTH - boxer["size"]
        )

        boxer["y"] = np.clip(
            boxer["y"],
            self.RING_TOP,
            self.RING_TOP + self.RING_HEIGHT - boxer["size"]
        )

        # punch
        if action in [5, 6, 7]:
            self.start_punch(boxer, action - 5) # punch offset

    # ---------------------------------------------------------
    # Collision check
    # ---------------------------------------------------------
    def _collide(self, a, b): # this must be upgraded such that it is checked if a punch land on the opponent 
        # using the player["hand"] position and updating the hand position based on the type of punch
        return (
            a["x"] < b["x"] + b["size"] and
            a["x"] + a["size"] > b["x"] and
            a["y"] < b["y"] + b["size"] and
            a["y"] + a["size"] > b["y"]
        )

    # ---------------------------------------------------------
    # Observation vector
    # ---------------------------------------------------------
    def _get_obs(self): # for now it is just like this
        return np.array([
            self.p1["x"], self.p1["y"], self.p1["stamina"], self.p1["state"],
            self.p2["x"], self.p2["y"], self.p2["stamina"], self.p2["state"],
        ], dtype=np.float32)

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------
    def old_render(self):
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # fill background
        self.screen.fill((220,220,220))

        def draw_ring():
            # draw ring border
            pygame.draw.rect(
                self.screen,
                (200, 200, 255),  # light bluish canvas
                (self.RING_LEFT, self.RING_TOP, self.RING_WIDTH, self.RING_HEIGHT)
            )

        draw_ring()

        def draw_boxer(b, color):
            # Draw body
            pygame.draw.rect(
                self.screen, color,
                (b["x"], b["y"], b["size"], b["size"])
            )

            # circle radius
            r = 10
            # base position
            lcx, lcy = b["x"] + b["size"] // 2, b["y"] + b["size"]
            rcx, rcy = b["x"] + b["size"] // 2, b["y"]

            offset_x, offset_y = 0, 0
            
            # Draw punch if active
            if b["state"] == 2 and b["punch"] is not None:
                
                # offset for each punch type
                if b["punch"] == 0:   # jab: forward
                    offset_x, offset_y = (30 if b == self.p1 else -30, 0)
                elif b["punch"] == 1: # hook: side swing
                    offset_x, offset_y = (40 if b == self.p1 else -40, 0)
                elif b["punch"] == 2: # uppercut: upwards
                    offset_x, offset_y = (50 if b == self.p1 else -50, 0)

            pygame.draw.circle(
                self.screen, color,
                (int(lcx + offset_x), int(lcy + offset_y)), r
            )
            pygame.draw.circle(
                self.screen, color,
                (int(rcx + offset_x), int(rcy + offset_y)), r
            )

        draw_boxer(self.p1, (0, 128, 255))
        draw_boxer(self.p2, (255, 100, 0))

        def draw_score():
            # example score display
            score_font = pygame.font.Font("fonts/PressStart2P.ttf", 28)
            divider_font = pygame.font.Font("fonts/PressStart2P.ttf", 22)

            score_p1 = score_font.render(str(self.p1['score']), True, (255,255,255))
            score_p2 = score_font.render(str(self.p2['score']), True, (255,255,255))

            divider = divider_font.render("|", True, (255,255,255))

            self.screen.blit(score_p1, (self.W//2 + 100, 20))
            self.screen.blit(divider, (self.W//2, 20))
            self.screen.blit(score_p2, (self.W//2 - 100, 20))

        draw_score()

        pygame.display.flip()
        self.clock.tick(60)

    def render(self):
        """
            This render needs 10 sprites to be defined:
            p1_idle.png
            p1_jab.png
            p1_hook.png
            p1_uppercut.png
            p2_idle.png
            p2_jab.png
            p2_hook.png
            p2_uppercut.png
        """
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((240, 240, 240))

        def draw_boxer(b, player_name):
            # Determine sprite
            if b["state"] == 0 or b["punch"] is None:
                sprite = self.sprites[player_name]["idle"]
            else:
                if b["punch"] == 0:
                    sprite = self.sprites[player_name]["jab"]
                elif b["punch"] == 1:
                    sprite = self.sprites[player_name]["hook"]
                elif b["punch"] == 2:
                    sprite = self.sprites[player_name]["uppercut"]
                else:
                    sprite = self.sprites[player_name]["idle"]

            # Draw sprite centered on the boxer's position
            sprite_rect = sprite.get_rect(center=(b["x"] + b["size"]//2, b["y"] + b["size"]//2))
            self.screen.blit(sprite, sprite_rect)

        draw_boxer(self.p1, "p1")
        draw_boxer(self.p2, "p2")

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

if __name__ == '__main__':
    
    env = BoxingEnv(render_mode="human")
    obs, info = env.reset()

    done = False

    while not done:
        a1 = env.action_space.sample()
        a2 = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(a1, a2)
        env.old_render()

    env.close()

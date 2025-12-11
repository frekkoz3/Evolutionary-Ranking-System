"""
    Developer : Bredariol Francesco

    boxer.py

    This file contain the implementation of the boxer class.
    These boxers are thought as simple square object that can land punches of rectangular shape.

    Class list:
        - Boxer
"""
import pygame 
import math
import numpy as np

SIZE = 40
SPEED = 5

class Boxer:
    """
    Represents a single boxer in the ring.
    Handles movement, punches, stamina, hitboxes, and animations.
    """
    def __init__(self, x, y, color, name="p1"):
        self.x = x
        self.y = y
        self.size = SIZE
        self.speed = SPEED
        self.color = color
        self.name = name
        self.score = 0
        self.sprite = None

        # Saving last position
        self.movement = (0, 0)

        # Combat stats
        self.stamina = 100
        self.max_stamina = 100

        # Punch state
        self.state = 0     # 0=idle, 1=startup, 2=active, 3=recovery
        self.timer = 0
        self.punch_type = None

        # Hitbox (active only during punch)
        self.hitbox = None

        # Punch definitions: (startup, active, recovery, stamina_cost)
        self.PUNCHES = {
            0: (5, 1, 5, 18),     # short punch
            1: (9, 1, 9, 15),  # mid punch
            2: (12, 1, 12, 10),  # long punch
        }

        # Storing last action and if it is punching
        self.is_punching = 0
        self.last_action = 0
    
    @classmethod
    def state_dim(self):
        return 28 # to tweak each time

    def get_state(self):
        """
            x, y, px, py, self.state, self.timer, self.stamina, self.speed, self.stamina_reg_lev(self.stamina), *(np.array(self.PUNCHES[0])*temp), *(np.array(self.PUNCHES[1])*temp), *(np.array(self.PUNCHES[2])*temp), self.is_punching, self.last_action, self.size//2, self.size//4, self.score
        """
        px, py = self.get_rect().center
        if self.hitbox != None:
            px, py = self.hitbox.center
        x, y = self.get_rect().center
        temp = np.array(self.punches_stamina_penalty())
        return x, y, *self.movement, px, py, self.state, self.timer, self.stamina, self.speed, self.stamina_reg_lev(self.stamina), *(np.array(self.PUNCHES[0])*temp), *(np.array(self.PUNCHES[1])*temp), *(np.array(self.PUNCHES[2])*temp), self.is_punching, self.last_action, self.size//2, self.size//4, self.score

    # ---------------------------------------------------------
    # Movement
    # ---------------------------------------------------------
    def move(self, action):
        self.last_action = action
        self.movement = (0, 0)
        self.speed = SPEED * self.stamina_reg_lev(min_value=0.2, max_value=1)
        if action == 0: # idle
            pass
        if action == 1:   # up
            self.y -= self.speed
            self.movement = (0, -1)
        elif action == 2: # down
            self.y += self.speed
            self.movement = (0, 1)
        elif action == 3: # left
            self.x -= self.speed
            self.movement = (-1, 0)
        elif action == 4: # right
            self.x += self.speed
            self.movement = (1, 0)

    # ---------------------------------------------------------
    # Start a punch
    # ---------------------------------------------------------

    def punches_stamina_penalty(self):
            q = 2
            m = - 1/100
            y = self.stamina * m + q
            return (y, 1, 1, y)
    
    def start_punch(self, punch_type):
        if self.state != 0:
            return
        startup, active, recovery, cost = tuple(np.array(self.PUNCHES[punch_type])*np.array(self.punches_stamina_penalty())) # we must multiply it by some penalty driven by the stamina level
        if self.stamina < cost:
            self.last_action = 0
            return -1
        
        self.is_punching = 1
        
        self.stamina -= cost
        self.state = 1
        self.timer = int(startup)
        self.punch_type = punch_type
        self.hitbox = None

    # ---------------------------------------------------------
    # Progress punch state
    # ---------------------------------------------------------
    def update_punch_state(self, facing_left=False):
        if self.state == 0:
            return

        self.timer -= 1

        _, active, recovery, _ = tuple(np.array(self.PUNCHES[self.punch_type])*self.punches_stamina_penalty())

        if self.state == 1 and self.timer <= 0:
            # startup → active
            self.state = 2
            self.timer = active

            # Create hitbox now
            self.make_hitbox(facing_left)

        elif self.state == 2 and self.timer <= 0:
            # active → recovery
            self.state = 3
            self.timer = int(recovery)
            self.hitbox = None

        elif self.state == 3 and self.timer <= 0:
            # recovery → idle
            self.state = 0
            self.is_punching = 0
            self.punch_type = None
            self.hitbox = None

    # ---------------------------------------------------------
    # Create punch hitbox
    # ---------------------------------------------------------
    def make_hitbox(self, facing_left=False):
        """
        Creates a hitbox rectangle in front of the boxer depending on punch type.
        """

        bx = self.x
        by = self.y
        s = self.size

        if self.punch_type == 0:   # short
            hx = bx - 20 if facing_left else bx + s
            w, h = 20, s // 2

        elif self.punch_type == 1: # mid
            hx = bx - 30 if facing_left else bx + s + 10
            w, h = 20, s//2

        elif self.punch_type == 2: # long
            hx = bx - 40 if facing_left else bx + s + 20
            w, h = 20, s//2

        hy = by + s // 2 if facing_left else by

        self.hitbox = pygame.Rect(hx, hy, w, h)

    # ---------------------------------------------------------
    # Punch cancelation
    # ---------------------------------------------------------
    def cancel_punch(self):
        self.state = 0
        self.timer = 0
        self.is_punching = 0
        self.punch_type = None
        self.hitbox = None

    # ---------------------------------------------------------
    # Boxer body rectangle
    # ---------------------------------------------------------
    #@property
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)

    # ---------------------------------------------------------
    # Regenerate stamina slowly using a sigmoid update based on the current level of stamina
    # ---------------------------------------------------------
    def stamina_reg_lev(self, min_value=0.05, max_value=0.1):
            """
            x: actual stamina level in [0, 100]
            min_value: output when x = 0
            max_value: output when x = 100
            """
            
            # Standard sigmoid centered at 50 with steepness k
            k = 0.1
            sigmoid = 1 / (1 + math.exp(-k * (self.stamina - 50)))

            # Normalize sig in [0,1] using its values at x=0 and x=100
            sig0 = 1 / (1 + math.exp(-k * (0 - 50)))
            sig100 = 1 / (1 + math.exp(-k * (100 - 50)))
            normalized = (sigmoid - sig0) / (sig100 - sig0)

            # Scale to [min_value, max_value]
            return min_value + normalized * (max_value - min_value)
    
    def regenerate(self):

        self.stamina = min(self.max_stamina, self.stamina + self.stamina_reg_lev())

    def __str__(self):
        s = f"{self.name}\n------\npos ({self.x}, {self.y})\nscore : {self.score}\nstamina : {self.stamina}\nstate : {self.state}\n------\n"
        return s

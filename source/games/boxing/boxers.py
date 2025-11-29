"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    boxer.py

    This file contain the implementation of the boxer class.
    These boxers are thought as simple square object that can land punches of rectangular shape.

    Class list:
        - Boxer
"""
import pygame 
import math

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

        # Storing last action
        self.last_action = 0
    
    @classmethod
    def state_dim(self):
        return 10 # to tweak each time

    def get_state(self):
        """
            x_center, y_center, punch_x_center, punch_y_center, state, stamina, last_action, half size, half punch size, score
        """
        px, py = self.get_rect().center
        if self.hitbox != None:
            px, py = self.hitbox.center
        x, y = self.get_rect().center

        return x, y, px, py, self.state, self.stamina, self.last_action, self.size//2, self.size//4, self.score

    # ---------------------------------------------------------
    # Movement
    # ---------------------------------------------------------
    def move(self, action):
        if action == 0: # idle
            self.last_action = action
        elif action == 1:   # up
            self.y -= self.speed
            self.last_action = action
        elif action == 2: # down
            self.y += self.speed
            self.last_action = action
        elif action == 3: # left
            self.x -= self.speed
            self.last_action = action
        elif action == 4: # right
            self.x += self.speed
            self.last_action = action

    # ---------------------------------------------------------
    # Start a punch
    # ---------------------------------------------------------
    def start_punch(self, punch_type):
        if self.state != 0:
            return
        
        self.last_action = punch_type + 5 # punch_type + 5 = action

        startup, active, recovery, cost = self.PUNCHES[punch_type]
        if self.stamina < cost:
            self.last_action = 0
            return -1
        
        self.stamina -= cost
        self.state = 1
        self.timer = startup
        self.punch_type = punch_type
        self.hitbox = None

    # ---------------------------------------------------------
    # Progress punch state
    # ---------------------------------------------------------
    def update_punch_state(self, facing_left=False):
        if self.state == 0:
            return

        self.timer -= 1

        if self.state == 1 and self.timer <= 0:
            # startup → active
            _, active, _, _ = self.PUNCHES[self.punch_type]
            self.state = 2
            self.timer = active

            # Create hitbox now
            self.make_hitbox(facing_left)

        elif self.state == 2 and self.timer <= 0:
            # active → recovery
            _, _, recovery, _ = self.PUNCHES[self.punch_type]
            self.state = 3
            self.timer = recovery
            self.hitbox = None

        elif self.state == 3 and self.timer <= 0:
            # recovery → idle
            self.state = 0
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
    def regenerate(self):

        def stamina(x, min_value=0.05, max_value=0.1):
            """
            x: actual stamina level in [0, 100]
            min_value: output when x = 0
            max_value: output when x = 100
            """
            # Standard sigmoid centered at 50 with steepness k
            k = 0.1
            sigmoid = 1 / (1 + math.exp(-k * (x - 50)))

            # Normalize sig in [0,1] using its values at x=0 and x=100
            sig0 = 1 / (1 + math.exp(-k * (0 - 50)))
            sig100 = 1 / (1 + math.exp(-k * (100 - 50)))
            normalized = (sigmoid - sig0) / (sig100 - sig0)

            # Scale to [min_value, max_value]
            return min_value + normalized * (max_value - min_value)
        
        self.stamina = min(self.max_stamina, self.stamina + stamina(self.stamina))

    def __str__(self):
        s = f"{self.name}\n------\npos ({self.x}, {self.y})\nscore : {self.score}\nstamina : {self.stamina}\nstate : {self.state}\n------\n"
        return s

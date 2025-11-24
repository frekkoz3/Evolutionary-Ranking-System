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

class Boxer:
    """
    Represents a single boxer in the ring.
    Handles movement, punches, stamina, hitboxes, and animations.
    """
    def __init__(self, x, y, color, name="p1"):
        self.x = x
        self.y = y
        self.size = 40
        self.speed = 5
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
            0: (5, 1, 8, 5),     # jab
            1: (10, 1, 14, 12),  # hook
            2: (14, 1, 18, 16),  # uppercut
        }

    # ---------------------------------------------------------
    # Movement
    # ---------------------------------------------------------
    def move(self, action):
        if action == 1:   # up
            self.y -= self.speed
        elif action == 2: # down
            self.y += self.speed
        elif action == 3: # left
            self.x -= self.speed
        elif action == 4: # right
            self.x += self.speed

    # ---------------------------------------------------------
    # Start a punch
    # ---------------------------------------------------------
    def start_punch(self, punch_type):
        if self.state != 0:
            return

        startup, active, recovery, cost = self.PUNCHES[punch_type]
        if self.stamina < cost:
            return

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

        if self.punch_type == 0:   # jab
            hx = bx - 10 if facing_left else bx + s
            hy = by + s // 4
            w, h = 20, s // 2

        elif self.punch_type == 1: # hook
            hx = bx - 5 if facing_left else bx + s - 15
            hy = by + 2
            w, h = 30, s - 4

        elif self.punch_type == 2: # uppercut
            hx = bx + s//3
            hy = by - 12
            w, h = s//2, 25

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
    # Regenerate stamina slowly
    # ---------------------------------------------------------
    def regenerate(self):
        self.stamina = min(self.max_stamina, self.stamina + 0.2)

    def __str__(self):
        s = f"{self.name}\n------\npos ({self.x}, {self.y})\nscore : {self.score}\nstamina : {self.stamina}\nstate : {self.state}\n------\n"
        return s

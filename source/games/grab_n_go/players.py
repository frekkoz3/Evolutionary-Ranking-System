"""
    Final Project for the "Optimization for AI" course.
    Developer : Bredariol Francesco

    player.py

    This file contain the player for the game "grab n go".

    Class list:
        - Player
"""
import pygame 
import math
import numpy as np

SIZE = 40
SPEED = 10

class Player():

    def __init__(self, x, y, name, role):
        self.x = x
        self.y = y
        self.last_x = x
        self.last_y = y
        self.speed = SPEED
        self.size = SIZE
        self.stamina = 100
        self.max_stamina = 100
        self.last_action = 0
        self.score = 0
        self.name = name
        self.role = role # it can be catcher or runner

    def move(self, action):
        self.last_x, self.last_y = self.get_rect().center
        self.last_action = action
        self.speed = SPEED
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

    def get_rect(self):
        return pygame.Rect(self.x, self.y, SIZE, SIZE)
    
    def get_color(self):
        return (255, 0, 0) if self.role == 'catcher' else (0, 255, 0)
    
    @classmethod
    def state_dim(self):
        return 7
    
    def get_state(self):
        return *self.get_rect().center, self.last_x, self.last_y, self.last_action, self.size//2, self.speed
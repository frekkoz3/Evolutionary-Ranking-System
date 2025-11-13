import numpy as np
import sys
import time

class PongEnv:
    def __init__(self, width=800, height=600, paddle_height=100, paddle_speed=6,
                 ball_speed=5, speedup_factor=1.05, randomness=0.15):
        # Environment configuration
        self.WIDTH = width
        self.HEIGHT = height
        self.PADDLE_HEIGHT = paddle_height
        self.PADDLE_SPEED = paddle_speed
        self.BALL_SPEED = ball_speed
        self.SPEEDUP_FACTOR = speedup_factor  # how much faster the ball gets each hit
        self.RANDOMNESS = randomness          # how much noise in ball motion
        self.timestep = 0
        self.reset()

    def reset(self, done = False):
        """Reset environment to initial state"""
        self.ball_x = self.WIDTH / 2
        self.ball_y = self.HEIGHT / 2

        # Random initial direction
        angle = np.random.uniform(-0.3, 0.3)
        direction = np.random.choice([-1, 1])
        self.vel_x = direction * self.BALL_SPEED * np.cos(angle)
        self.vel_y = self.BALL_SPEED * np.sin(angle)

        # Paddles and scores
        self.paddle_a_y = self.HEIGHT / 2
        self.paddle_b_y = self.HEIGHT / 2
        if done:
            self.score_a = 0
            self.score_b = 0

        self.timestep = 0 

        return self._get_state()

    def _get_state(self):
        """Return normalized state"""
        return np.array([
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.vel_x / self.BALL_SPEED,
            self.vel_y / self.BALL_SPEED,
            self.paddle_a_y / self.HEIGHT,
            self.paddle_b_y / self.HEIGHT
        ], dtype=np.float32)

    def step(self, action_a, action_b):
        """Perform one simulation step.
        action_a, action_b ∈ {-1, 0, 1}: up, stay, down
        """
        # To check if one of the player scored
        scored = False
        # To check if one of the player won
        done = False

        self.timestep += 1

        reward_a, reward_b, done = 0, 0, False

        # === Paddle control ===
        # Introduce lag: paddle moves only every n steps
        if self.timestep % 3 == 0:
            self.paddle_a_y += action_a * self.PADDLE_SPEED
            self.paddle_b_y += action_b * self.PADDLE_SPEED

        # Clamp paddles to play area
        self.paddle_a_y = np.clip(self.paddle_a_y, self.PADDLE_HEIGHT/2, self.HEIGHT - self.PADDLE_HEIGHT/2)
        self.paddle_b_y = np.clip(self.paddle_b_y, self.PADDLE_HEIGHT/2, self.HEIGHT - self.PADDLE_HEIGHT/2)

        # === Ball movement ===
        self.ball_x += self.vel_x
        self.ball_y += self.vel_y

        # Top and bottom bounce
        if self.ball_y <= 0 or self.ball_y >= self.HEIGHT:
            self.vel_y *= -1

        # === Paddle collisions with spin + acceleration ===
        # Left paddle
        if (self.ball_x <= 30 and abs(self.paddle_a_y - self.ball_y) < self.PADDLE_HEIGHT / 2):
            offset = (self.ball_y - self.paddle_a_y) / (self.PADDLE_HEIGHT / 2)
            self.vel_x = abs(self.vel_x) * self.SPEEDUP_FACTOR
            self.vel_y = (offset * self.BALL_SPEED) * self.SPEEDUP_FACTOR
            reward_a += 0.1

        # Right paddle
        elif (self.ball_x >= self.WIDTH - 30 and abs(self.paddle_b_y - self.ball_y) < self.PADDLE_HEIGHT / 2):
            offset = (self.ball_y - self.paddle_b_y) / (self.PADDLE_HEIGHT / 2)
            self.vel_x = -abs(self.vel_x) * self.SPEEDUP_FACTOR
            self.vel_y = (offset * self.BALL_SPEED) * self.SPEEDUP_FACTOR
            reward_b += 0.1

        # === Slight random noise in physics ===
        self.vel_x += np.random.uniform(-self.RANDOMNESS, self.RANDOMNESS)
        self.vel_y += np.random.uniform(-self.RANDOMNESS, self.RANDOMNESS)

        # Normalize velocity to prevent runaway speeds
        speed = np.sqrt(self.vel_x**2 + self.vel_y**2)
        max_speed = self.BALL_SPEED * 3.0
        if speed > max_speed:
            self.vel_x *= (max_speed / speed)
            self.vel_y *= (max_speed / speed)

        # === Scoring ===
        if self.ball_x <= 0:
            self.score_b += 1
            reward_b += 1
            scored = True
            
        elif self.ball_x >= self.WIDTH:
            self.score_a += 1
            reward_a += 1
            scored = True
        
        if self.score_a == 5 or self.score_b == 5:
            done = True

        # Adding some random noise 
        return self._get_state() + np.random.normal(0, 0.1, size=6), (reward_a, reward_b), done, scored, {}

    def render_ascii(self, grid_w=60, grid_h=20, clear_screen=True):
            """Render ASCII field with borders, paddles, and ball"""
            # Clear terminal (fast ANSI method)
            if clear_screen:
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()

            grid = [[" "] * grid_w for _ in range(grid_h)]

            # Ball position
            bx = int(self.ball_x / self.WIDTH * (grid_w - 1))
            by = int(self.ball_y / self.HEIGHT * (grid_h - 1))

            # Paddle positions in grid coordinates
            paddle_a_center = int(self.paddle_a_y / self.HEIGHT * grid_h)
            paddle_b_center = int(self.paddle_b_y / self.HEIGHT * grid_h)
            paddle_size = max(1, int(self.PADDLE_HEIGHT / self.HEIGHT * grid_h / 2))

            # Draw paddles
            for py in range(paddle_a_center - paddle_size, paddle_a_center + paddle_size + 1):
                if 0 <= py < grid_h:
                    grid[py][0] = "|"
            for py in range(paddle_b_center - paddle_size, paddle_b_center + paddle_size + 1):
                if 0 <= py < grid_h:
                    grid[py][-1] = "|"

            # Draw ball
            if 0 <= by < grid_h and 0 <= bx < grid_w:
                grid[by][bx] = "O"

            # Borders
            top_border = "#" * (grid_w + 2)
            bottom_border = "#" * (grid_w + 2)

            # Combine
            lines = [top_border]
            for row in grid:
                lines.append("#" + "".join(row) + "#")
            lines.append(bottom_border)

            # Print frame
            print("\n".join(lines))
            print(f"Score: A={self.score_a} | B={self.score_b}")

def play()

# === Example usage ===
if __name__ == "__main__":
    env = PongEnv()
    env.reset(done = True)

    while True:
        # Simple AI: paddles track ball position
        action_a = np.sign(env.ball_y - env.paddle_a_y)
        action_b = np.sign(env.ball_y - env.paddle_b_y)

        state, (r_a, r_b), done, scored, _ = env.step(action_a, action_b)

        #env.render_ascii()

        #time.sleep(0.05)  # Adjust speed for smoother animation

        if scored:
            #time.sleep(0.5)
            env.reset(done)

        if done:
            time.sleep(0.5)
            print("Game finished: ")
            break
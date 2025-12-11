# GRAB N GO

This folder contains a custom implementation of a simplified "Grab And Go" game as a Gymnasium-compatible environment, designed for both human play and AI agent experimentation.

---

## Structure

```bash
grab_n_go        # Folder containing the implementation of a variant of the "grab andd go" game as gymnasium environment
├── players.py    # Python implementation of the "player" class
└── grab_n_go.py    # Python implementation of the "grab_n_go" environment
```

---

## Features

- **Two-player environment:** Supports two agents (or a human vs. agent) on the ground.
- **Actions:** Move.
- **Hitboxes:** Collisions are handled via rectangular hitboxes for each player.
- **Scoring:** Catcher wins if he catches the runner under 30 seconds. Runner wins otherwise.
- **Rendering:** Human-friendly visualization using Pygame.
- **Gymnasium API:** Fully compatible with standard Gymnasium workflows (`env.reset()`, `env.step()`, `env.render()`).

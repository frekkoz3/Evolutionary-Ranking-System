# Boxing 2600 (Variant)

This folder contains a custom implementation of a simplified "Boxing 2600" game as a Gymnasium-compatible environment, designed for both human play and AI agent experimentation.

---

## Structure

boxing           # Folder containing the implementation of a variant of the "boxing 2600" game as gymnasium environment
├── fonts        # Folder containing the fonts for the human rendering of the game
├── sprites      # Folder containing the sprites for the human rendering of the game
├── boxers.py    # Python implementation of the "boxer" class
└── boxing.py    # Python implementation of the "boxing" environment

---

## Features

- **Two-player environment:** Supports two agents (or a human vs. agent) in the ring.
- **Actions:** Move, punch or other configurable actions.
- **Hitboxes:** Collisions are handled via rectangular hitboxes for each boxer.
- **Scoring:** Points are tracked for each boxer during a match.
- **Rendering:** Human-friendly visualization using Pygame with sprites and fonts.
- **Gymnasium API:** Fully compatible with standard Gymnasium workflows (`env.reset()`, `env.step()`, `env.render()`).

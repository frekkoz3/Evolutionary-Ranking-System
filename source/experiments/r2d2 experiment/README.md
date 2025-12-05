# 🧪 R2D2 EXPERIMENT

This folder contains an improved version of the Boxing experiment, where the classical Deep Q-Network (DQN) is replaced by the Recurrent Replay Distributed DQN (R2D2) algorithm.

R2D2 enhances DQN by introducing recurrent neural networks (RNNs), sequence-based replay, and burn-in for hidden-state stabilization.\
It is particularly effective in partially observable environments, such as Boxing, where the instantaneous observation does not fully describe the game state.

---

## 📌 Why R2D2?

DQN suffers from one major limitation:

> ✔ It assumes full observability — that a single state captures all relevant information.

In Boxing (and many real environments), this is false:

- Opponent decisions depend on momentum or recent movement
- Some cues are not directly observable in every frame
- Temporal patterns matter
- Single-step frames → insufficient to infer full environment state

R2D2 improves upon DQN by:

| Feature                          | Benefit                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| **Recurrent network (LSTM/GRU)** | Keeps memory of past states, enabling partial observability handling |
| **Sequence-level replay buffer** | Stores full trajectories instead of isolated transitions             |
| **Burn-in phase**                | Stabilizes hidden states before computing TD targets                 |
| **Better credit assignment**     | Longer temporal dependencies can be learned naturally                |
| **More stable training**         | Because RNN state is restored during replay                          |

## 📌 Description of the Environment

(Implemented in games/boxing)

- Game: Boxing (Atari 2600)
- Interface: Gym / ALE
- State: Preprocessed observations from the custom Boxing environment
- Action space: Discrete  actions (move, punch)
- Rewards: Based on striking, dodging, scoring, and win/loss outcomes

The environment remains exactly the same as in the DQN experiment.
Only the learning algorithm is improved.

---

## 📌 R2D2 Implementation Overview

R2D2 follows the core structure of DQN but introduces key extensions.

1. Recurrent Q-Network

    - Architecture: feature encoder → LSTM/GRU → linear Q-value head
    - Keeps hidden state across time steps
    - Learns through sequences, not isolated frames

2. Recurrent Replay Buffer

    Instead of single transitions (s, a, r, s'), the buffer stores:

    ```scss
        (sequence of states, actions, rewards, dones)
    ```

    This allows:
    - Training on temporally coherent trajectories
    - Proper handling of hidden states

3. Burn-In Period

    When replaying a sequence:

    - The first k steps only update the hidden state
    - No Q-loss is computed until after burn-in

    This ensures the RNN begins the training sequence in a realistic hidden state.

## 📌 Project Structure

```bash
r2d2_experiment/
│
├── replay_buffer.py          # Sequence-based replay memory with burn-in
├── train.py                  # Full R2D2 training loop
├── evaluate.py               # Evaluation episodes using recurrent model
└── README.md                 # This file
```

## 📌 How to Run

1. Set up the virtual environment:

    ```console
    .../Evolutionary-Ranking-System/.venv/Scripts/activate
    ```

2. Start training:

    ```console
    .../Evolutionary-Ranking-System py -m source.experiments.r2d2_experiment.train
    ```

3. Evaluate:

    Please, referes to the evaluate.py file to understand the possible arguments.

    ```console
    .../Evolutionary-Ranking-System py -m source.experiments.r2d2_experiment.evaluate 
    ```

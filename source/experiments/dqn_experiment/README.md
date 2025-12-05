# 🧪 DQN EXPERIMENT

This folder contains a minimal implementation of a Deep Q-Network (DQN) learning to play the Atari 2600 game Boxing.\
The goal of this experiment is to demonstrate how a reinforcement learning agent can learn competitive behavior against a very simple built-in logical opponent.\
I'll follow the DQN implementation presented by this [Pytorch's tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

---

## 📌 Description of the Environment

(The environment is implemented in the games/boxing folder)

- **Game**: Boxing (Atari 2600)
- **Interface**: Gym / ALE
- **State**: Preprocessed information from the game
- **Action space**: Discrete agent actions (move, punch)
- **Reward**: various reward based on hitting, winning, dodging and so on

---

## 📌 The Opponent (Logical Bot)

The agent is trained against a deterministic rule-based bot, which performs simple behavior such as:

- Attacking if close to the player
- Retreating if far behind
- Searching for the opponent

This opponent does not learn, and exists only to provide a stable training target for the DQN agent.

---

## 📌 DQN Implementation Overview

The agent follows the standard DQN training pipeline:

1. **Neural Network**
    - Input : state
    - Output : Q-values for each possible action
2. **Replay buffer**
    - Stores transitions *(state, action, reward, next_state, done)* to decorrelate training samples
3. **$\epsilon$-Greedy policy**
    - Starts with high exploration
    - Gradually decays $\epsilon$ to favor exploitation
4. **Target Newtork**
    - A copy of the main network
    - Updated slowly for stable training
5. **Optimization**
    - Temporal Difference objective: $$L = (r + \lambda max_{a'} Q_{target}(s', a') - Q(s, a))^2$$

---

## 📌 Project Structure

```bash
dqn_experiment/
│
├── replay_buffer.py      # Circular experience memory
├── train.py              # Full training loop
├── evaluate.py           # Run evaluation episodes
└── README.md (this file)
```

---

## 📌 How to Run

1. Set up the virtual environment:

    ```console
    .../Evolutionary-Ranking-System/.venv/Scripts/activate
    ```

2. Start training:

    ```console
    .../Evolutionary-Ranking-System py -m source.experiments.dqn_experiment.train
    ```

3. Evaluate:

    Please, referes to the evaluate.py file to understand the possible arguments.

    ```console
    .../Evolutionary-Ranking-System py -m source.experiments.dqn_experiment.evaluate 
    ```

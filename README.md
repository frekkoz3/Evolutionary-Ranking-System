# 🧬​ EVOLUTIONARY RANKING SYSTEM

This is the repo for the final project of the "Optimization for AI" course.

The goal of this project is to create a frawmework for a novable optimization method for zero-sum games (and for all problems that can be map into a zero-sum game).\
The main idea is to use the already existing ELO system (based not on a in-game fitness function but on a statistical off-game fitness function) in order to generate individuals capable of playing always better and better (until the reach the global optimum that we could call "meta" in this scenario).\
In addiction this system add some extra components by evolutionary strategies, incorporating mutations and crossovers, in order to leverage the exploration (the ELO system should works as a leverage for the exploitation).\
In the end propers ranks will be added in order to create different "level of difficulty" (such as bronze, silver, gold etc) in order to make the progression for new individual more gradual (since each rank should be increasingly more difficult to reach since individuals populating a rank should be stronger and stronger).

This system want to combine two types of optimization: **ONLINE OPTIMIZATION** (based on Reinforcement Learning or any other online learner algorithm) and **OFFLINE OPTIMIZATION** (based on Evolutionary Strategy). The idea is to use the ELO as a link between these two optimization techniques.

---

To begin with we must talk about what is the ELO.

## ♟️​ ELO

> The Elo (ranking) system is a method for calculating the relative skill levels of players in zero-sum games such as chess or esports. (...) The difference in the ratings between two players serves as a predictor of the outcome of a match. Two players with equal ratings who play against each other are expected to score an equal number of wins. A player whose rating is 100 points greater than their opponent's is expected to score 64%; if the difference is 200 points, then the expected score for the stronger player is 76%.

from [*Wikipedia*](https://en.wikipedia.org/wiki/Elo_rating_system)

Whenever an individual plays against an other one it can gain or lose ELO points based on the match's result and in the prior success' probability.

The ELO ranking system is something that must be tuned and it is not a really easy task.\
For that, during this project, I will use a simple version of it defined as follows:

1. Starting with a population of $n$ individuals, each of them will have an uniform probability of winning against each other. This means that $ELO(x) = c \forall x \in Population$, where $c \in \mathbb{N}$ is a positive integer.

2. When competing individuals will gain and lose point based on the relative probability of winning/loosing. This probability will be something proportional to ratio of the ELO of two individuals. The actual proposal for this function will be: $Gain(x, y) = \alpha \frac{ELO(y)}{ELO(x)}$. Note that the Loss function is the same just with the inverted sign.

3. At the end of the match update the ELO of each individual based on the actual GAIN or LOSS.

---

Now let's talk about individuals.

## 👾​ INDIVIDUALS

Inidivduals can be essentialy whatever.\
The main idea is that individuals must follows the Individual Protocol descripted in the **'individual.py'** file.\
Individuals may differs based on the benchmark. For example, in the Atari world it has seens to use individuals that are in fact RL algorithm, while one can think of GP individuals for other benchmarks.

Individual must be able to perform some mutations (even trivial ones) and possibly crossover (even if this is not mandatory).

Before putting individuals in the *evolutionary ranking system framework* one should work on them in order to be sure they work.

---

### 📌​ Structure

```bash
Evolutionary-Ranking-System
│
├── literature               # Folder containing some papers regarding the argument
├── source                   # Folder containing the code
    ├── agents               # Folder for the agents
        ├── dqn_agent        # Folder containing the dqn agent implementation
        ├── gng_dqn_agent    # Folder containing the dqn agent implementation for the grab and go environment
        ├── gng_tree_agent   # Folder containing the tree agent implementation for the grab and go environment
        ├── r2d2_agent       # Folder containing the r2d2 agent implementation
        ├── tree_agent       # Folder containing the tree agent implementation
        └── individual.py    # Python implementation of the individual class
    ├── debug                # Folder for the debugging
        └── profiles         # Folder containing debugging profiles of the code
    
    ├── elo_system           # Folder containing the elo system
        ├── elo_system.py    # Python implementation of the wrapper for the entire project
        ├── ELO.py           # Python implementation of the ELO
        ├── evo_utils.py     # Some utils
        └── matchmaking.py   # Python implementation of the matchmaking system

    ├── experiments          # Folder containing somee experiments
        ├── dqn_exp          # Folder containing an experimentation of the dqn algorithm
        └── r2d2_exp         # Folder containing an experimentation on the r2d2 algorithm - NOT IMPLEMENTED

    ├── games                # Folder containing the implementation of some desired games as gymnasium environment
        ├── console.py       # Python implementation of the handler between agents and environments
        ├── grab_n_go.py     # Folder containing the implementation of the "Grab And Go" game as gymnasium environment
        └── boxing           # Folder containing the implementation of a variant of the "boxing 2600" game as gymnasium environment

    ├── individuals          # Folder containing individuals files

    └── main.py              # Main function to call

├── tree_png                 # Folder containing tree-policies representation (used for tree agents)
├── README.md (this file)
├── requirements.txt         # Requirements for the virtual environment
└── todo.md                  # File containing the remaing things to develop
```

---

### 📌​ How to use

This is the simplest example of usage of the system.

1. Set up the virtual environment:

    ```console
    .../Evolutionary-Ranking-System/.venv/Scripts/activate
    ```

2. Install requirements.txt:

    ```console
    .../Evolutionary-Ranking-System/pip install -r requirements.txt
    ```

3. Run the main:

    ```console
    .../Evolutionary-Ranking-System py -m source.main --test True
    ```

    The possible flags for the main are:

    ```console
        --test (bool) : test mode (just visualize results for the grab 'n go benchmark with tree agent (GP))
        --train (bool) : train mode (train individuals using the configuration saved in the config/config.json for the grab 'n go benchmark with tree agent (GP))
        --help (bool) : help (resent you here)
    ```

# 🌳 Tree-Based Genetic Programming Agent

This folder contains a flexible evolutionary agent using symbolic tree-based policies.

---

## 📌 Overview

This module implements a tree-based genetic programming (GP) agent..

Each agent maintains several expression trees (one per possible action).\
At inference time, each tree produces a numerical score and the agent selects:

```bash
action = argmax(tree_outputs)
```

Trees evolve through mutation and a reward-driven online adaptation mechanism.

---

## 🧠 Architecture

### Tree Representation

A tree node is a simple Python dict.

Terminal Node:

```python
{
  "type": "terminal",
  "name": "x"                # feature name
}
```

Constant terminal:

```python
{
  "type": "terminal",
  "name": "const",
  "value": 3.14
}
```

Function node:

```python
{
  "type": "func",
  "op": "add",
  "children": [child1, child2]
}
```

---

## Mutation

Mutation happens in 2 steps:

1. Select which subtree to mutate
2. Select which type of mutation to apply to that subtree

Mutation is controlled by:

- subtree_prob – probability of selectin each subtree
- prob_subtree – probability of replacing a subtree
- prob_node – probability of mutating a function node
- prob_const – probability of mutating a terminal

---

## Reward-driven online adaptation mechanism

Since this GP does not deploy any mechanism of the traditional policy optimization (neither RL neither LCS), a simple heuristic is applied in order to try to help the mutation process.

When a reward is collected:

- if the reward is positive : decrease the probability of mutating the subtree that receives that reward
- if the reward is negative : increase the probability of mutating the subtree that receives that penalty

In addiction to this an update probability has been added. An agent update (mutate) with this probability.

When a reward is collected:

- if the reward is positive : decrease the probability of updating
- if the reward is negative : increase the probability of updating

---

## 🔍 Visualization

Make sure Graphviz is installed and available on PATH.

This tree implementation makes it possible to visualize the policy structure via png. Once you use the individual.visualize() method you can find the png in the tree folder.

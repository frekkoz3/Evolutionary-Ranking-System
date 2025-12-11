"""
    Developer : Bredariol Francesco

    tree.py

    This file contains the implementation of an tree agent.
"""

import copy
import random
import math
import pickle
import os
import numpy as np
from source.agents.individual import Individual
import graphviz
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz/bin" # please change to the location of your graphviz installation
from source import TREE_DIR

MAX_DEPTH = 6  # maximum depth for generated subtrees
MAX_NODES = 30  # cap to avoid runaway bloat

# --- Operators and arities ---
OP_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "min": 2,
    "max": 2,
    "abs": 1,
    "gt": 2,    # 1 if a > b else 0
    "lt": 2,    # 1 if a < b else 0
    "eq": 2,    # 1 if a == b else 0
    "if": 3     # cond, true_branch, false_branch
}

# Operators that are safe to choose for numeric outputs
NUMERIC_FUNCS = ["add", "sub", "mul", "min", "max", "abs", "sin", "cos"]

# Full allowed set (includes comparators and if)
ALLOWED_FUNCS = list(OP_ARITY.keys())

# Default terminals (features) 
DEFAULT_FEATURES = [
    "x", "y", 
    "dx", "dy",
    "remaining_x", "remaining_y"
]

CONST_RANGE = 20 # Range factor for constant generation
CONST_SCALE = 1 # Scale factor for constant mutation

CONST_PROB = 0.3 # Probability of selecting a costant instead over a feature

LR = 0.2 # The learning rate is in fact how much to increase or decrease the probability of mutation of a given subtree given positive or negative reward (positive : X(1-LR/2), negative : X(1+LR))
INITIAL_UPDATE_PROB = 0.5 # This is the initial probability of updating
MIN_UPDATE_PROB = 0.05 # This is the minimum probability of updating
MAX_UPDATE_PROB = 0.9 # This is the maximum probability of updating
EPS = 0.005 # This is how much the probabiity of updating changes each time a reward is collected ()

class TreeAgent(Individual):
    """
    Tree-based policy individual.
    Representation:
      - Each tree is a nested dict node.
      - Terminal node: {"type":"terminal", "name": <feature_name or "const">, "value": <if const>}
      - Func node: {"type":"func", "op": <op_name>, "children": [child_nodes...]}
    Output:
      - This individual contains n_trees (default 5) each producing a numeric score; the action is argmax(scores).
    """

    def __init__(self, init_elo=100, n_trees=5, features=None, seed=None, **kwargs):
        super().__init__(init_elo)
        if seed is not None:
            random.seed(seed)

        self.n_trees = n_trees
        self.trees_prob = np.ones(self.n_trees) * 1/n_trees
        self.features = features if features is not None else DEFAULT_FEATURES
        self.trees = [None for _ in range(self.n_trees)]
        self.to_update = False
        self.build_random_policy()
        self.update_prob = INITIAL_UPDATE_PROB

    def need_map(self):
        return True

    # -----------------------------
    #  Tree construction & utils
    # -----------------------------
    def random_terminal(self):
        """Return a terminal node: either a feature or a constant."""
        if random.random() < 1 - CONST_PROB and len(self.features) > 0:
            name = random.choice(self.features)
            return {"type": "terminal", "name": name}
        else:
            # constant terminal
            return {"type": "terminal", "name": "const", "value": random.uniform(-CONST_RANGE, CONST_RANGE)}

    def random_func_node(self, depth):
        """Create a function node with random op and children."""
        # bias towards numeric functions at deeper depths to avoid boolean-only nodes
        op = random.choice(ALLOWED_FUNCS)
        arity = OP_ARITY[op]
        children = [self.random_tree(depth + 1) for _ in range(arity)]
        return {"type": "func", "op": op, "children": children}

    def random_tree(self, depth=0):
        """
        Ramped random tree:
          - If depth >= MAX_DEPTH -> terminal
          - Else choose terminal with some probability proportional to the depth {1 - exp[(2*depth)/MAX_DEPTH]}
        """
        if depth >= MAX_DEPTH or random.random() < (1 - math.exp(-(2*depth/MAX_DEPTH))): # with this prob depth must be at least > 1
            return self.random_terminal()
        else:
            return self.random_func_node(depth)

    def build_random_policy(self):
        """Initialize all output trees with random trees."""
        self.trees = [self.random_tree(depth=0) for _ in range(self.n_trees)]
        self._ensure_size_limits()

    # -----------------------------
    #  Evaluation
    # -----------------------------
    def evaluate_tree(self, node, obs):
        """
        Evaluate node given observation dict obs (mapping feature names -> floats).
        Comparators return 1.0 or 0.0. IF checks cond > 0.5 to decide true branch.
        """
        t = node["type"]
        if t == "terminal":
            if node["name"] == "const":
                return float(node.get("value", 0.0))
            # Feature lookup: missing features default to 0.0
            return float(obs.get(node["name"], 0.0))

        # func node
        op = node["op"]
        children = node["children"]

        if op == "add":
            return self.evaluate_tree(children[0], obs) + self.evaluate_tree(children[1], obs)
        if op == "sub":
            return self.evaluate_tree(children[0], obs) - self.evaluate_tree(children[1], obs)
        if op == "mul":
            return self.evaluate_tree(children[0], obs) * self.evaluate_tree(children[1], obs)
        if op == "sin":
            return math.sin(self.evaluate_tree(children[0], obs))
        if op == "cos":
            return math.cos(self.evaluate_tree(children[0], obs))
        if op == "min":
            return min(self.evaluate_tree(children[0], obs), self.evaluate_tree(children[1], obs))
        if op == "max":
            return max(self.evaluate_tree(children[0], obs), self.evaluate_tree(children[1], obs))
        if op == "abs":
            return abs(self.evaluate_tree(children[0], obs))
        if op == "gt":
            return 1 if self.evaluate_tree(children[0], obs) > self.evaluate_tree(children[1], obs) else 0
        if op == "lt":
            return 1 if self.evaluate_tree(children[0], obs) < self.evaluate_tree(children[1], obs) else 0
        if op == "eq":
            return 1 if self.evaluate_tree(children[0], obs) == self.evaluate_tree(children[1], obs) else 0
        if op == "if":
            cond_val = self.evaluate_tree(children[0], obs)
            # treat any cond_val > 0.5 as true
            if cond_val > 0.5:
                return self.evaluate_tree(children[1], obs)
            else:
                return self.evaluate_tree(children[2], obs)

        # fallback
        return 0.0
    
    # --------------------------------
    # Simil-online-learning
    # --------------------------------
    def observe(self, obs, action, rew, new_obs, done, **kwargs):
        """
            This is a simple heuristic for computing which subtree leads to the wrong / right decision
        """
        # WE COULD ACTUALLY THING AT SOMETHING INTELLIGENT HERE BUT WE HAVE NO TIME SO LET THIS AS IT IS
        if rew > 0: # we just assume positive reward = good 
            self.trees_prob[action] *= (1 - LR/2)
            self.trees_prob.clip(min=0.1/self.n_trees, max = 0.9)
            self.trees_prob /= np.sum(self.trees_prob)
            self.update_prob = max(MIN_UPDATE_PROB, self.update_prob - EPS*rew) 
        if rew < 0:
            self.trees_prob[action] *= (1 + LR)
            self.trees_prob.clip(min=0.1/self.n_trees, max = 0.9)
            self.trees_prob /= np.sum(self.trees_prob)
            self.update_prob = min(MAX_UPDATE_PROB, self.update_prob - EPS*rew) 
        if done: self.to_update = True

    def update(self, **kwrags):
        if self.to_update:
            self.mutate(mutation_prob=self.update_prob, prob_subtree=0.05, prob_node=0.45, prob_const=0.45)
            self.to_update = False

    def move(self, obs=None, eval_mode = False, **kwargs):
        """Returns action id (0..n_trees-1). If obs is None fallback to 0."""
        if obs is None:
            return 0
        # Evaluate each tree and pick highest scoring action
        scores = [self.evaluate_tree(t, obs) for t in self.trees]
        # break ties deterministically (argmax with index)
        best_idx = max(range(self.n_trees), key=lambda i: (scores[i], -i))
        return best_idx

    # -----------------------------
    #  Tree utilities (paths)
    # -----------------------------
    def _get_all_paths(self, node):
        """
        Return a list of paths (each path is a list of child indices) to every node in the tree.
        The root path is [].
        """
        paths = []

        def walk(n, path):
            paths.append(path.copy())
            if n["type"] == "func":
                for i, ch in enumerate(n["children"]):
                    path.append(i)
                    walk(ch, path)
                    path.pop()

        walk(node, [])
        return paths

    def _get_node_by_path(self, root, path):
        """Return node at path (list of indices). If path == [] -> root."""
        n = root
        for idx in path:
            n = n["children"][idx]
        return n

    def _set_node_by_path(self, root, path, new_node):
        """Replace node at path with new_node."""
        if path == []:
            return new_node
        parent_path = path[:-1]
        parent = self._get_node_by_path(root, parent_path)
        parent["children"][path[-1]] = new_node
        return root

    def tree_size(self, root):
        """Return number of nodes in tree."""
        return len(self._get_all_paths(root))

    def _ensure_size_limits(self):
        """Ensure all trees are within MAX_NODES; if not, shrink by subtree mutation."""
        for i in range(len(self.trees)):
            size = self.tree_size(self.trees[i])
            if size > MAX_NODES:
                # replace with smaller random tree
                self.trees[i] = self.random_tree(depth=0)

    # -----------------------------
    #  Mutation operators
    # -----------------------------
    def mutate(self, mutation_prob = 0.5, prob_subtree=0.4, prob_node=0.4, prob_const=0.2):
        """High-level mutation: choose a mutation type by probability."""
        r = random.random()
        if r < mutation_prob:
            r = random.random()
            if r < prob_subtree:
                self.mutate_subtree()
            elif r < prob_subtree + prob_node:
                self.mutate_node_operator()
            else:
                self.mutate_constant_or_terminal()
            self._ensure_size_limits()

    def mutate_subtree(self):
        """Pick a random tree and replace a random subtree with a new random subtree."""
        ti = np.random.choice(np.arange(0, self.n_trees), p = self.trees_prob)
        root = self.trees[ti]
        paths = self._get_all_paths(root)
        # chose one random path within all the possible paths
        chosen_path = random.choice(paths)
        new_sub = self.random_tree(depth=0)
        if chosen_path == []: # root
            self.trees[ti] = new_sub
        else:
            self._set_node_by_path(self.trees[ti], chosen_path, new_sub)

    def mutate_node_operator(self):
        """
        Change an operator at a random func node to another operator with compatible arity,
        or change a terminal variable to another feature.
        """
        ti = np.random.choice(np.arange(0, self.n_trees), p = self.trees_prob)
        root = self.trees[ti]
        paths = self._get_all_paths(root)
        # filter only func nodes (non-terminals)
        func_paths = [p for p in paths if self._get_node_by_path(root, p)["type"] == "func"]
        if not func_paths:
            # fallback: mutate terminal instead
            self.mutate_constant_or_terminal()
            return
        chosen = random.choice(func_paths)
        node = self._get_node_by_path(root, chosen)
        current_op = node["op"]
        arity = OP_ARITY[current_op]
        # choose new op with same arity (prefer numeric funcs)
        candidates = [op for op, a in OP_ARITY.items() if a == arity and op != current_op]
        if candidates:
            new_op = random.choice(candidates)
            node["op"] = new_op
            # if new op arity differs (shouldn't), we would need to adjust children; kept safe by same arity filter
        else:
            # nothing to change; small chance to change a child instead
            idx = random.randrange(len(node["children"]))
            node["children"][idx] = self.random_tree(depth=0)

    def mutate_constant_or_terminal(self):
        """Either change a constant value slightly or swap a terminal feature."""
        ti = np.random.choice(np.arange(0, self.n_trees), p = self.trees_prob)
        root = self.trees[ti]
        paths = self._get_all_paths(root)
        # choose a terminal node path (including const)
        term_paths = [p for p in paths if self._get_node_by_path(root, p)["type"] == "terminal"]
        if not term_paths:
            # fallback: subtree mutation
            self.mutate_subtree()
            return
        chosen = random.choice(term_paths)
        node = self._get_node_by_path(root, chosen)
        if node["name"] == "const":
            # small gaussian perturbation
            node["value"] += random.gauss(0, CONST_SCALE)
        else:
            # swap to another feature or to a constant
            if random.random() < CONST_PROB:
                node["name"] = "const"
                node["value"] = random.uniform(-CONST_RANGE, CONST_RANGE)
            else:
                node["name"] = random.choice(self.features)

    # -----------------------------
    #  Crossover (not really used but since it is usefull we keep it)
    # -----------------------------
    def crossover(self, other):
        """
        Classical subtree crossover:
          - pick random tree index in self and other (they share same action indexing)
          - pick random subtree in each, swap
        Returns a new TreeIndividual (child).
        """
        child = copy.deepcopy(self)
        # action-tree index
        ti = random.randrange(self.n_trees)
        a_root = child.trees[ti]
        b_root = copy.deepcopy(other.trees[ti])

        a_paths = self._get_all_paths(a_root)
        b_paths = self._get_all_paths(b_root)

        a_choice = random.choice(a_paths)
        b_choice = random.choice(b_paths)

        # extract subtree from b_root at b_choice
        subtree_b = self._get_node_by_path(b_root, b_choice)
        # place it into child (a_root) at a_choice
        if a_choice == []:
            child.trees[ti] = copy.deepcopy(subtree_b)
        else:
            child._set_node_by_path(child.trees[ti], a_choice, copy.deepcopy(subtree_b))

        child._ensure_size_limits()
        return child

    def save(self, path):
        d = os.path.dirname(path) 
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    # -----------------------------
    #  Utilities
    # -----------------------------
    def _node_to_str(self, node, depth=0):
        indent = "  " * depth
        if node["type"] == "terminal":
            if node["name"] == "const":
                return f"{indent}{node['value']:.3f}"
            return f"{indent}{node['name']}"
        s = f"{indent}{node['op']}"
        for ch in node["children"]:
            s += "\n" + self._node_to_str(ch, depth + 1)
        return s

    def tree_str(self, ti=0):
        if ti < 0 or ti >= self.n_trees:
            return "<invalid tree index>"
        return self._node_to_str(self.trees[ti])

    def __repr__(self):
        return f"<TreeIndividual id={self.id} elo={self.elo} trees={self.n_trees}>"
    
    def export_tree_dot(self, ti=0):
        """
        Return a graphviz.Digraph object representing tree index ti.
        """
        if ti < 0 or ti >= self.n_trees:
            raise ValueError("Invalid tree index")

        root = self.trees[ti]
        dot = graphviz.Digraph(comment=f"Tree {ti}")
        dot.attr("node", shape="box", style="filled", color="#DDEEFF")

        # We'll assign a unique ID to each node (incremental counter)
        counter = [0]

        def add_node(node):
            nid = f"n{counter[0]}"
            counter[0] += 1

            if node["type"] == "terminal":
                if node["name"] == "const":
                    label = f"Const\n{node['value']:.2f}"
                else:
                    label = node["name"]
                dot.node(nid, label)
            else:
                op = node["op"]
                dot.node(nid, op)
                for child in node["children"]:
                    cid = add_node(child)
                    dot.edge(nid, cid)

            return nid

        add_node(root)
        return dot

    def visualize_tree(self, ti = 0, filename = "tree"):
        dot = self.export_tree_dot(ti)
        outpath = dot.render(os.path.join(TREE_DIR, filename), format="png", cleanup=True)

    def visualize(self, filename = "tree"):
        for i in range (self.n_trees):
            self.visualize_tree(ti = i, filename=f"{i}_{filename}")

    def _reset_probs(self):
        self.trees_prob = np.ones(self.n_trees)/self.n_trees
        self.update_prob = INITIAL_UPDATE_PROB
    
if __name__ == "__main__":

    random_states = [{f : random.randint(-10, 10) for f in DEFAULT_FEATURES} for i in range (10)]
    t = TreeAgent()
    for i, random_state in enumerate(random_states):
        print(f"{i} : {t.move(random_state)}")
    t.mutate()
    for i, random_state in enumerate(random_states):
        print(f"{i} : {t.move(random_state)}")

    t.visualize()
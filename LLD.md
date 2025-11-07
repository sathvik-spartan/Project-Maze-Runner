# Low Level Design — Q-Learning vs Policy Iteration for GridWorld

## 1. Purpose

The purpose of this system is to implement a deterministic GridWorld Markov Decision Process (MDP), solve it using two different approaches:

* **Policy Iteration** (Dynamic Programming)
* **Tabular Q-Learning** (Model-Free Reinforcement Learning)

and compare:

* learned policies
* value functions
* learning behavior (rewards across episodes)

The output must allow side-by-side comparison and also must export metrics and visuals useful for performance evaluation.

---

## 2. Design Overview

The system consists of modular components:

| Component        | Role                                                   |
| ---------------- | ------------------------------------------------------ |
| Environment      | Implements GridWorld state transition dynamics         |
| Policy Iteration | Solves MDP via dynamic programming                     |
| Q-Learning       | Optimizes action values via online exploration         |
| Analysis         | Plots learning curves, compares policies, exports CSVs |
| Configuration    | Stores hyperparameters and environment specs           |

This allows running repeatable experiments, swapping settings, and evaluating convergence.

---

## 3. Environment LLD (GridWorld)

### Class: `GridWorld`

| Responsibility                                        |
| ----------------------------------------------------- |
| maintain grid, obstacles, start & terminal states     |
| implement transition function + rewards               |
| provide a list of all reachable states                |
| support both deterministic and stochastic transitions |

### Key Methods

| Method                           | Description                                          |
| -------------------------------- | ---------------------------------------------------- |
| `reset()`                        | resets to start state                                |
| `step(action)`                   | applies action, returns `(next_state, reward, done)` |
| `states()`                       | returns all valid states (tuples)                    |
| `is_terminal(state)`             | checks goal                                          |
| `transition_from(state, action)` | model-based transition (no side effects)             |
| `render_policy(policy)`          | prints arrows per cell for visualization             |

### Data Structures

* state is tuple `(row, col)`
* actions are integer indices mapped to `(dx, dy)`
* obstacles stored as a `set` for O(1) membership

---

## 4. Policy Iteration LLD

### Functions

| Function                                       | Description                     |
| ---------------------------------------------- | ------------------------------- |
| `policy_evaluation(env, policy, gamma, theta)` | iterative evaluation of Vπ      |
| `policy_iteration(env, gamma, max_iters)`      | runs full policy iteration loop |

### Notes

* terminal states never improved
* use synchronous updates
* improvement step uses greedy action based on current V

---

## 5. Q-Learning LLD

### Function

| Function                                                               | Description                  |
| ---------------------------------------------------------------------- | ---------------------------- |
| `q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, ...)` | runs tabular Q-learning loop |

### Behavior

* epsilon-greedy exploration strategy
* `defaultdict(np.zeros(...))` for Q
* returns:

  * Q table
  * greedy policy derived from Q
  * value function V(s)=max_a Q(s,a)
  * per-episode rewards for plotting

---

## 6. Analysis LLD

### Responsibilities

| Task                                           |
| ---------------------------------------------- |
| plot reward curves vs episodes                 |
| moving average computation                     |
| export final policies to CSV                   |
| textual visualization of policies in grid form |

### Output artifacts (typical)

* `learning_curve.png`
* `policies_comparison.csv`
* `summary_comparison.csv`

---

## 7. Configuration LLD

A configuration class must centralize:

* grid size, start cell, goal cell, obstacles
* default reward & goal reward
* Q-learning hyperparameters

This ensures reproducibility and easy experiment re-runs.

---

## 8. Sequence Flow

1. Load config
2. Create GridWorld env
3. Run Policy Iteration → get π_PI, V_PI
4. Run Q-Learning → get Q, π_Q, V_Q, rewards list
5. Pass results to analysis module
6. Display policies side-by-side
7. Save plots and CSVs
8. Present summary table comparing both methods

---

## 9. Testing Strategy

| Test Target      | Test                                                         |
| ---------------- | ------------------------------------------------------------ |
| environment      | boundary stepping, obstacle collision behavior               |
| policy iteration | 2x2 small grid → expected known policy                       |
| q-learning       | check Q converges to near-optimal policy after long episodes |
| analysis         | CSVs created, plots saved, no crashes                        |

---

## 10. Scalability Considerations

| Method           | Scaling                                                             |   |                                            |
| ---------------- | ------------------------------------------------------------------- | - | ------------------------------------------ |
| Policy Iteration | grows O(                                                            | S | ×A) per sweep → expensive with large grids |
| Q-Learning       | sample based, can run forever → more flexible, slower to optimality |   |                                            |

Memory footprint: both maintain O(|S|×A) tables.

---

## 11. Future Extensions

* SARSA, Expected-SARSA, DQN
* stochastic slip probabilities
* animations of trajectories
* GUI or Web UI for interactive exploration

---

# Q-Learning vs Policy Iteration — GridWorld Navigation

## 1. Problem Description

This project implements a classic Markov Decision Process (MDP) GridWorld environment, and solves it using two different Reinforcement Learning approaches:

| Method               | Type                | Access to Environment Model? |
| -------------------- | ------------------- | ---------------------------- |
| **Policy Iteration** | Dynamic Programming | Yes — full model             |
| **Q-Learning**       | Model-Free RL       | No — learns by interaction   |

Goal: analyze how planning (Policy Iteration) compares to learning from experience (Q-Learning) on the same environment.

The agent starts in a fixed start cell and must navigate to a goal cell while avoiding obstacles and accruing penalty for each step (negative default reward).

We compare:

* final policies
* value functions
* learning curves over training episodes
* convergence behavior

---

## 2. Methodology

### 2.1 Environment

GridWorld is deterministic (optional stochastic slip can be added).
Each state is a cell `(row, col)` in a discrete grid.

Reward structure:

| Transition    | Reward                                             |
| ------------- | -------------------------------------------------- |
| every step    | small negative reward (to encourage shortest path) |
| reaching goal | +1 reward                                          |
| obstacles     | impassable ((of an obstacle) Incapable of being overcome or surmounted. ) |

---

### 2.2 Policy Iteration

Policy Iteration performs:

1. **Policy Evaluation** → compute V<sup>π</sup> until convergence
2. **Policy Improvement** → greedify policy from current V
3. repeat until stable

Outputs:
optimal deterministic policy π*, optimal value function V*.

---

### 2.3 Q-Learning

Q-Learning learns Q(s,a) via Temporal Difference update:

```
Q[s,a] ← Q[s,a] + α * (r + γ * max_a' Q[s',a'] − Q[s,a])
```

Decision making: ε-greedy during learning, greedy during evaluation.

Outputs:
final Q table, greedy policy derived from Q, reward curve across episodes.

---

## 3. Implementation

This project is implemented as **one single Google Colab notebook** for reproducibility.

Notebook includes:

* environment class
* policy iteration implementation
* Q learning implementation
* plotting, CSV export, evaluation metrics

---

## 4. How to Run (Colab)

### Step 1: Open Google Colab

[https://colab.research.google.com/](https://colab.research.google.com/)

### Step 2: Upload the Notebook

Upload the `.ipynb` provided with this project.

### Step 3: Run All Cells in Order

`Runtime → Run all`

The notebook will:

* define the GridWorld
* run policy iteration
* run Q-learning
* plot reward curves
* print and compare both learned policies side-by-side

All results are produced automatically.

---

## 5. Expected Outputs

* final policy from Policy Iteration (text grid arrows)
* final policy from Q-Learning (text grid arrows)
* learning curve plot (reward vs episodes)
* comparison CSVs (optional)

Example qualitative outcome:

| Method           | Converges To Optimal?     | Needs Model? | Learns Online? |
| ---------------- | ------------------------- | ------------ | -------------- |
| Policy Iteration | Yes                       | Yes          | No             |
| Q-Learning       | Yes (after many episodes) | No           | Yes            |

---

## 6. Conclusion

| Observation                     | Interpretation                                               |
| ------------------------------- | ------------------------------------------------------------ |
| Policy Iteration converges fast | because it performs full dynamic programming using the model |
| Q-Learning converges gradually  | because it must explore and estimate values via sampling     |

Final takeaway:

> Both methods reach similar optimal policy, but via very different philosophies.
> Policy Iteration **plans**, Q-Learning **learns**.

---

## 7. References

* Sutton & Barto — *Reinforcement Learning: An Introduction*
* OpenAI Gym / Tabular RL baselines literature

---
## 8. Additional References 
- [Github Reference](https://github.com/MJeremy2017/reinforcement-learning-implementation)
- [Medium Blog Post 1](https://medium.com/data-science/implement-grid-world-with-q-learning-51151747b455)
- [Medium Blog Post 2 - Implementation from Scratch](https://medium.com/@zhangyue9306/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff)
- [Medium Blog Post 2 - continuation](https://medium.com/@zhangyue9306/implement-grid-world-with-q-learning-51151747b455)

---

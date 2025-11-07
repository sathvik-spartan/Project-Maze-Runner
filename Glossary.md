## Deterministic vs Non-Deterministic Environments (Reinforcement Learning)

### Deterministic Environment

* **Definition:**
  An environment is deterministic if taking an action from a given state always leads to the *same* next state.
* **Example:**
  If the agent chooses `UP`, it always moves exactly one cell upward (unless blocked).
* **Mathematical View:**

  $$
  P(s' \mid s, a) = 1 \quad \text{for one specific next state } s'
  $$
* **Meaning in code:**
  `DETERMINISTIC = True` → The agent always reaches the intended next state.

---

### Non-Deterministic (Stochastic) Environment

* **Definition:**
  An environment is non-deterministic if the result of an action is uncertain, and it may lead to different next states with certain probabilities.
* **Example:**
  If the agent chooses `UP`, there might be a 70% chance to move up, but a 30% chance to slip left or right.
* **Mathematical View:**

  $$
  P(s' \mid s, a) < 1 \quad \text{for multiple possible next states}
  $$
* **Meaning in code:**
  `DETERMINISTIC = False` → The agent’s action might randomly end up doing something else.

---

Briefly:

* **Deterministic = predictable** (action → guaranteed outcome).
* **Non-Deterministic = unpredictable** (action → probabilistic outcome).

---

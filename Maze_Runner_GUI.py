# -*- coding: utf-8 -*-
"""
Maze_Runner_GUI.py
Author: Kailash Khadarbad

A complete Tkinter GUI for visualizing Policy Iteration and Q-Learning
on a 6x6 GridWorld Maze environment.
"""

# ---------------------------- Imports ----------------------------
import numpy as np
import random
from collections import defaultdict
from typing import Tuple, List, Dict
import tkinter as tk
from tkinter import messagebox
import time
import threading

# ---------------------------- GridWorld ----------------------------
class GridWorld:
    def __init__(self,
                 width:int=6,
                 height:int=6,
                 start:Tuple[int,int]=(0,0),
                 goal:Tuple[int,int]=(5,5),
                 obstacles:List[Tuple[int,int]]=None,
                 default_reward:float=-0.04,
                 goal_reward:float=1.0):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()
        self.default_reward = default_reward
        self.goal_reward = goal_reward
        self.actions = [(0,-1),(1,0),(0,1),(-1,0)]  # Up, Right, Down, Left
        self.action_names = ['U','R','D','L']
        self.reset()

    def in_bounds(self, s): 
        return 0 <= s[0] < self.width and 0 <= s[1] < self.height
    
    def is_terminal(self, s): 
        return s == self.goal

    def step(self, action:int):
        if self.is_terminal(self.state): 
            return self.state, 0.0, True
        dx, dy = self.actions[action]
        nx, ny = self.state[0]+dx, self.state[1]+dy
        ns = (nx, ny)
        if (not self.in_bounds(ns)) or (ns in self.obstacles):
            ns = self.state
        self.state = ns
        if self.is_terminal(ns):
            return ns, self.goal_reward, True
        return ns, self.default_reward, False

    def reset(self):
        self.state = self.start
        return self.state

    def states(self):
        return [(x,y) for y in range(self.height) for x in range(self.width)
                if (x,y) not in self.obstacles]

# ---------------------------- Policy Iteration ----------------------------
def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = {s:0.0 for s in env.states()}
    while True:
        delta = 0
        for s in env.states():
            if env.is_terminal(s): 
                continue
            a = policy[s]
            saved = env.state
            env.state = s
            ns, r, done = env.step(a)
            env.state = saved
            v = r + gamma * (0 if done else V[ns])
            delta = max(delta, abs(V[s]-v))
            V[s] = v
        if delta < theta: 
            break
    return V

def policy_iteration(env, gamma=0.99, max_iters=1000):
    policy = {s: random.randrange(len(env.actions)) for s in env.states() if not env.is_terminal(s)}
    for _ in range(max_iters):
        V = policy_evaluation(env, policy, gamma)
        stable = True
        for s in env.states():
            if env.is_terminal(s): 
                continue
            old_a = policy[s]
            best_a = old_a
            best_val = -1e9
            for a in range(len(env.actions)):
                saved = env.state
                env.state = s
                ns, r, done = env.step(a)
                env.state = saved
                val = r + gamma * (0 if done else V[ns])
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        if stable: 
            break
    return policy, V

# ---------------------------- Q-Learning ----------------------------
def q_learning(env, episodes=3000, alpha=0.6, gamma=0.99,
               epsilon=0.3, epsilon_decay=0.9996, max_steps=200, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    Q = defaultdict(lambda: np.zeros(len(env.actions)))
    rewards = []
    success = 0
    for ep in range(episodes):
        s = env.reset()
        total = 0
        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                a = np.random.randint(len(env.actions))
            else:
                a = int(np.argmax(Q[s]))
            ns, r, done = env.step(a)
            total += r
            Q[s][a] += alpha * (r + gamma * (0 if done else np.max(Q[ns])) - Q[s][a])
            s = ns
            if done:
                if env.is_terminal(s): 
                    success += 1
                break
        rewards.append(total)
        epsilon *= epsilon_decay

    policy = {s: int(np.argmax(Q[s])) for s in env.states() if not env.is_terminal(s)}
    V = {s: np.max(Q[s]) for s in policy}
    return Q, policy, V, rewards, success

# ---------------------------- Tkinter GUI ----------------------------
class MazeRunnerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Maze Runner - Reinforcement Learning Demo")
        self.cell_size = 70

        # Initialize environment
        self.env = GridWorld(width=6, height=6,
                             start=(0,0), goal=(5,5),
                             obstacles=[(1,1),(2,1),(3,1),(3,2),(3,3),(1,4)])
        
        # Canvas for grid display
        self.canvas = tk.Canvas(master,
                                width=self.env.width*self.cell_size,
                                height=self.env.height*self.cell_size,
                                bg="white")
        self.canvas.pack(side="left", padx=10, pady=10)

        # Controls frame
        controls = tk.Frame(master)
        controls.pack(side="right", padx=10, pady=10)

        tk.Button(controls, text="Run Policy Iteration", command=self.run_policy_iteration, width=20).pack(pady=5)
        tk.Button(controls, text="Run Q-Learning", command=self.run_q_learning, width=20).pack(pady=5)
        tk.Button(controls, text="Animate Agent", command=self.animate_agent, width=20).pack(pady=5)
        tk.Button(controls, text="Reset", command=self.reset_env, width=20).pack(pady=5)
        tk.Button(controls, text="Exit", command=master.quit, width=20).pack(pady=5)

        self.status = tk.Label(controls, text="Status: Ready", fg="blue")
        self.status.pack(pady=10)

        self.policy = None
        self.agent_id = None
        self.draw_grid()

    # Draw maze grid
    def draw_grid(self):
        self.canvas.delete("all")
        for y in range(self.env.height):
            for x in range(self.env.width):
                x1, y1 = x*self.cell_size, y*self.cell_size
                x2, y2 = x1+self.cell_size, y1+self.cell_size
                s = (x, y)
                color = "lightblue"
                if s == self.env.start: color = "limegreen"
                elif s == self.env.goal: color = "red"
                elif s in self.env.obstacles: color = "black"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="white")

                # Draw arrows for policy
                if self.policy and s in self.policy and s != self.env.goal and s not in self.env.obstacles:
                    a = self.env.action_names[self.policy[s]]
                    arrow = {"U":"↑","R":"→","D":"↓","L":"←"}[a]
                    self.canvas.create_text(x1+self.cell_size/2, y1+self.cell_size/2,
                                            text=arrow, font=("Arial", 18, "bold"))

    # Run Policy Iteration
    def run_policy_iteration(self):
        self.status.config(text="Running Policy Iteration...")
        self.master.update()
        self.policy, _ = policy_iteration(self.env)
        self.status.config(text="Policy Iteration Completed!")
        self.draw_grid()

    # Run Q-Learning (threaded)
    def run_q_learning(self):
        self.status.config(text="Running Q-Learning... (Please wait)")
        self.master.update()
        threading.Thread(target=self._run_q_learning_thread).start()

    def _run_q_learning_thread(self):
        _, self.policy, _, _, success = q_learning(self.env, episodes=3000, seed=42)
        self.status.config(text=f"Q-Learning Completed! Successes: {success}")
        self.draw_grid()

    # Animate learned policy
    def animate_agent(self):
        if not self.policy:
            messagebox.showwarning("No Policy", "Run Policy Iteration or Q-Learning first!")
            return

        self.env.reset()
        s = self.env.start
        self.agent_id = self.canvas.create_oval(5, 5, self.cell_size-5, self.cell_size-5, fill="yellow")
        self.master.update()

        for _ in range(100):
            if self.env.is_terminal(s): break
            a = self.policy.get(s)
            if a is None: break
            ns, _, _ = self.env.step(a)
            self.move_agent(s, ns)
            s = ns
            time.sleep(0.3)
        self.status.config(text="Animation Finished!")

    def move_agent(self, s, ns):
        x, y = s
        nx, ny = ns
        dx = (nx - x) * self.cell_size
        dy = (ny - y) * self.cell_size
        self.canvas.move(self.agent_id, dx, dy)
        self.master.update()

    def reset_env(self):
        self.policy = None
        self.env.reset()
        self.draw_grid()
        self.status.config(text="Environment Reset!")

# ---------------------------- Run the App ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MazeRunnerGUI(root)
    root.mainloop()

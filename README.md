# Snake RL Agent

A Reinforcement Learning based Snake AI built from scratch using Python, PyTorch, and Pygame.

The agent learns to play the classic Snake game using Deep Q-Learning with a handcrafted 14-dimensional state representation instead of image-based CNN input.

---

# Demo

> Gameplay GIF

```text
assets/gameplay.gif
```

---

# Project Overview

This project implements a Deep Q-Learning agent capable of learning efficient gameplay strategies in the classic Snake environment.

Unlike many implementations that rely on image processing or convolutional neural networks, this agent operates using a compact handcrafted state vector describing:

* Immediate collision danger
* Two-step collision danger
* Current movement direction
* Relative food position

The result is a lightweight and fast-learning reinforcement learning system capable of achieving high scores at high simulation speeds.

---

# Features

* Deep Q-Learning implementation from scratch
* Custom Snake environment using Pygame
* Experience Replay Memory
* Epsilon-Greedy Exploration Strategy
* Checkpoint Saving & Resume Training
* Fast training environment
* Human-playable Snake mode
* Handcrafted feature engineering instead of CNNs
* Lightweight neural network architecture

---

# Tech Stack

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Core Programming Language |
| PyTorch    | Neural Network & Training |
| Pygame     | Game Environment          |
| NumPy      | Numerical Operations      |
| Matplotlib | Training Visualization    |

---

# Reinforcement Learning Approach

The agent uses:

* Deep Q-Learning (DQN-inspired)
* Experience Replay
* Bellman Equation based Q-value updates
* Epsilon-Greedy exploration

The model predicts Q-values for 3 possible actions:

```text
[ Straight , Right Turn , Left Turn ]
```

---

# State Representation

The environment state is represented using a handcrafted 14-dimensional feature vector.

## Components

### Danger Detection

The agent checks for:

* Immediate danger straight
* Immediate danger right
* Immediate danger left
* Two-step danger straight
* Two-step danger right
* Two-step danger left

### Direction Encoding

Current movement direction:

* Moving Left
* Moving Right
* Moving Up
* Moving Down

### Food Position Encoding

Relative food location:

* Food Left
* Food Right
* Food Up
* Food Down

---

# Neural Network Architecture

The neural network architecture:

```text
Input Layer:   14
Hidden Layer:  256
Output Layer:  3
```

Architecture Flow:

```text
14 → 256 → ReLU → 3
```

The output layer predicts Q-values for each possible action.

---

# Reward System

| Event             | Reward |
| ----------------- | ------ |
| Eating Food       | +10    |
| Collision / Death | -10    |
| Normal Step       | -0.05  |

This reward shaping encourages:

* Survival
* Food collection
* Efficient pathfinding
* Reduced idle looping

---

# Training Strategy

## Exploration vs Exploitation

The agent follows an epsilon-greedy strategy:

```python
epsilon = 150 - n_games
```

Initially:

* More random exploration

Later:

* More model-based exploitation

---

# Experience Replay

Replay memory size:

```python
MAX_MEMORY = 100_000
```

Mini-batch training size:

```python
BATCH_SIZE = 1000
```

This improves:

* Stability
* Generalization
* Training efficiency

---

# Project Structure

```text
snake-ai-reinforcement-learning/
│
├── agent.py
├── model.py
├── snake_game_env.py
├── snake_human.py
├── helper.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── model.pth
│   ├── checkpoint.pth
│   └── agent_state.pth
│
├── assets/
│   └── gameplay.gif
│
└── .gitignore
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/roshan1285/snake-ai-reinforcement-learning.git
cd snake-ai-reinforcement-learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the AI Training

```bash
python agent.py
```

---

# Play the Human Version

```bash
python snake_human.py
```

---

# Training Results

> Replace these placeholders with your actual metrics.

| Metric                 | Value    |
| ---------------------- | -------- |
| Highest Score Achieved | 102       |
| Training Games         | 762     |
| Training Time          | 0.5 Hours |
| Average Stable Score   | 50       |

---

# Learning Highlights

This project was built to understand:

* Reinforcement Learning fundamentals
* Q-Learning mechanics
* Neural Network training pipelines
* Experience Replay systems
* Reward engineering
* Feature engineering for RL
* Environment-agent interaction
* Game AI architecture

---

# Future Improvements

Potential upgrades:

* Double DQN
* Prioritized Experience Replay
* Target Networks
* TensorBoard logging
* GPU acceleration
* CNN-based visual state learning
* Multi-agent environments
* Dynamic difficulty scaling

---

# Status

Project Status:

```text
Active / Experimental
```

The project is still being improved and optimized.

---

# Author

Roshan Savani

AI / ML / Software Development

GitHub:

```text
https://github.com/roshan1285/
```

---

# License

This project is open-source and available under the MIT License.

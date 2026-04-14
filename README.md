# Deep Q-Learning for Atari Enduro
### Comparison of DQN, DDQN, and Vectorized Variants

This project evaluates four Deep Q-Learning approaches on the Atari game **Enduro**, using PyTorch and the Gymnasium ALE environment:

- **DQN**
- **Double DQN (DDQN)**
- **Vectorized DQN (DQN\_Vec)** – 4 parallel environments
- **Vectorized DDQN (DDQN\_Vec)** – 4 parallel environments

The goal is to study how **double Q-learning** and **environment vectorization** affect:
- Sample efficiency  
- Training stability  
- Final performance  
- Wall-clock training time  

---

## Abstract

We investigate the role of algorithmic improvements (DDQN) and system-level improvements (vectorized environments) in Deep Q-Learning. Vectorized agents leverage multiple parallel environments—using `AsyncVectorEnv`—to reduce temporal correlation and accelerate learning. All models use classic Atari preprocessing: grayscale, resize to 84×84, frame stacking, and action repeat.

---

## Methods

### Implemented Agents
- **DQN** – classic Deep Q-Network
- **DDQN** – removes overestimation bias by decoupling action selection & evaluation
- **Vectorized DQN/ DDQN** – use **4 parallel environments** to generate diverse experience faster

### Core RL Components
- Replay Buffer  
- Target Network (periodically updated)  
- Adam optimizer (`lr = 1e-4`)  
- Mini-batches of 32–64  
- Discount factor `γ = 0.99`  
- Standard FrameStack, GrayScale, Resize  

### Vectorized Setup
We use:
```python
from gymnasium.vector import AsyncVectorEnv
```

### Results:
1. DDQN:

https://github.com/user-attachments/assets/c5ea4368-ae02-4552-8ec1-67a0eb99fabd

2. DDQN_Vec

https://github.com/user-attachments/assets/3db09f75-4c79-420e-bb54-a6143b8b2978

3. DQN

https://github.com/user-attachments/assets/818a5c4a-b98f-4026-87e2-b8c98fb87f71

4. DQN_Vec

https://github.com/user-attachments/assets/506195d0-3f7e-45a1-990a-492835397d8d

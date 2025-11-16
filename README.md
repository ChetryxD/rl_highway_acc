# Reinforcement Learning for Adaptive Cruise Control (ACC)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-brightgreen)](https://gymnasium.farama.org/)
[![RL](https://img.shields.io/badge/RL-DQN-orange)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

This project implements a **Reinforcement Learning–based Adaptive Cruise Control (ACC)** system in a simplified 1D highway scenario.

A **DQN agent** learns to follow a lead vehicle while:

- Maintaining a **safe time headway**
- Avoiding **collisions**
- Keeping control inputs **comfortable** (low accelerations)

The core of the project is a **custom Gymnasium environment** plus a clean, reproducible RL training + evaluation pipeline.

---

## 🔍 High-Level Overview

- **Environment:** 1D highway with an ego vehicle and a stochastic lead vehicle  
- **State representation:**
  - Ego speed [m/s]  
  - Lead speed [m/s]  
  - Distance gap [m]  
  - Relative speed = lead speed − ego speed  
- **Action space (discrete):**
  - 0 → strong brake (min deceleration)  
  - 1 → mild brake  
  - 2 → hold speed (0 accel)  
  - 3 → mild acceleration  
  - 4 → strong acceleration (max accel)  
- **Reward design:**
  - Quadratic penalty on deviation from **desired time headway**
  - Penalty on large acceleration (comfort)
  - Strong penalty on **collision**
  - Small penalty when **overspeeding**

The goal is to learn a **policy** that balances safety and comfort purely from interaction.

---

## 🧱 Project Structure

```text
rl_highway_acc/
├─ acc_env/
│  ├─ __init__.py              # Exposes HighwayACCEnv
│  └─ highway_acc_env.py       # Custom Gymnasium environment
│
├─ configs/
│  └─ default_config.yaml      # DQN hyperparameters & training settings
│
├─ scripts/
│  ├─ test_env_run.py          # Sanity-check: run env with random policy
│  ├─ train_dqn.py             # Train DQN agent on HighwayACCEnv
│  └─ evaluate_policy.py       # Evaluate a trained model and print rewards
│
├─ models/                     # Saved models (ignored by git)
├─ notebooks/                  # For future analysis / plots (optional)
├─ tests/                      # For future unit tests (optional)
│
├─ environment.yml             # Conda environment specification
├─ .gitignore
└─ README.md
Environment & Agent Design
HighwayACCEnv (Gymnasium)

The environment models longitudinal motion only:

Ego & lead vehicle speeds are integrated using simple kinematics.

The lead vehicle has a random acceleration component at each step.

The distance gap is updated as:
### Distance Gap Update

The longitudinal gap is updated using simple kinematics:

$$
\text{gap}_{t+1}
=
\text{gap}_t
+
(\text{lead}_v - \text{ego}_v) \cdot \Delta t
$$

### Desired Time Headway

The desired gap follows a time-headway rule:

$$
\text{desired\_gap} = T_{\text{headway}} \cdot \text{ego\_speed}
$$



A time headway policy is used to define the desired gap:

desired_gap
=
𝑇
headway
⋅
ego_speed
desired_gap=T
headway
	​

⋅ego_speed

The reward combines:

Headway error: r_gap = -k_gap * (gap - desired_gap)^2

Comfort: r_comfort = -k_acc * accel^2

Overspeed penalty

Collision penalty (large negative reward + episode termination)

DQN Agent

Implemented using Stable-Baselines3

Policy: MlpPolicy (fully-connected neural network)

Core hyperparameters are stored in configs/default_config.yaml:

learning_rate

buffer_size

batch_size

gamma

exploration_fraction, exploration_final_eps

total_timesteps, etc.

The training script wraps the custom environment in a Monitor and trains DQN end-to-end, saving the final model into models/

⚙️ Setup
# Clone the repository
git clone https://github.com/ChetryxD/rl_highway_acc.git
cd rl_highway_acc

# Create and activate the conda environment
conda env create -f environment.yml
conda activate rl_highway_acc
▶️ Usage

1. Sanity-check the environment (random actions)python scripts/test_env_run.py
This runs a few steps with a random policy and prints the state + reward:t=  0.1s | ego_v=15.85 m/s | lead_v=19.31 m/s | gap= 39.60 m
Step 0 | reward=...
...
2. Train the DQN agent
Hyperparameters are defined in configs/default_config.yaml.
To start training:
python scripts/train_dqn.py

This will:

Create a models/ folder (if not present)

Train DQN for total_timesteps

Save the model as:models/dqn_highway_acc_YYYYMMDD-HHMMSS.zip
3. Evaluate a trained policy
Edit the model path in scripts/evaluate_policy.py: model_path = os.path.join("models", "dqn_highway_acc_YYYYMMDD-HHMMSS")

Then run: python scripts/evaluate_policy.py
You’ll see per-step logs and per-episode returns, e.g.:  === Episode 1 ===
t=  0.1s | ego_v=... | lead_v=... | gap=...
...
Episode 1 total reward: -1732.45
...
Mean reward over 3 episodes: ...

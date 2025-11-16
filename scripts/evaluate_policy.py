# scripts/evaluate_policy.py

import os
import sys
import numpy as np

from stable_baselines3 import DQN

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acc_env import HighwayACCEnv  # type: ignore


def evaluate(model_path: str, n_episodes: int = 5, render: bool = True):
    env = HighwayACCEnv(render_mode="human")
    model = DQN.load(model_path)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        print(f"\n=== Episode {ep + 1} ===")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()

        episode_rewards.append(ep_reward)
        print(f"Episode {ep + 1} total reward: {ep_reward:.2f}")

    env.close()

    print("\n==========================")
    print(f"Mean reward over {n_episodes} episodes: {np.mean(episode_rewards):.2f}")
    print("==========================")


if __name__ == "__main__":
    # Set this to your actual trained model filename (without .zip)
    # Example: models/dqn_highway_acc_20251116-201530.zip
    model_path = os.path.join("models", "dqn_highway_acc")
    evaluate(model_path, n_episodes=3, render=True)

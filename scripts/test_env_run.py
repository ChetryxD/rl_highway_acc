# scripts/test_env_run.py

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acc_env import HighwayACCEnv


def main():
    env = HighwayACCEnv(render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", obs)

    for t in range(40):
        action = env.action_space.sample()  # random discrete action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {t} | reward={reward:.3f} | done={done}")
        if done:
            print("Episode ended early.")
            break

    env.close()


if __name__ == "__main__":
    main()

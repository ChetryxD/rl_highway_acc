import os
import sys
import yaml
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Make sure Python can see the project root so we can import acc_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acc_env import HighwayACCEnv  # type: ignore


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env():
    """
    Factory function for the environment.
    SB3 expects a callable that returns a new env instance.
    """
    env = HighwayACCEnv()
    # Monitor wraps the env to record episode stats (useful later)
    env = Monitor(env)
    return env


def main():
    # 1. Load config
    config_path = os.path.join("configs", "default_config.yaml")
    config = load_config(config_path)

    # 2. Create vectorized env (even with 1 env, SB3 prefers this)
    vec_env = make_vec_env(make_env, n_envs=1)

    # 3. Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 4. Initialize DQN model with config parameters
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        train_freq=config["train_freq"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        verbose=1,
    )

    total_timesteps = config["total_timesteps"]

    print(f"Starting training for {total_timesteps} timesteps...")

    # 5. Train
    model.learn(total_timesteps=total_timesteps)

    # 6. Save final model (simple, stable name)
    save_path = os.path.join(models_dir, "dqn_highway_acc")
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")


if __name__ == "__main__":
    main()

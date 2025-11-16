# acc_env/highway_acc_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HighwayACCEnv(gym.Env):
    """
    Simple 1D highway Adaptive Cruise Control environment.

    State:
        [ego_speed, lead_speed, distance_gap, relative_speed]

    Action (DISCRETE):
        0: strong brake      -> min_accel  (e.g. -6.0 m/s^2)
        1: mild brake        -> -3.0 m/s^2
        2: no acceleration   -> 0.0 m/s^2
        3: mild acceleration -> 1.5 m/s^2
        4: strong accel      -> max_accel  (e.g. 3.0 m/s^2)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, dt=0.1):
        super().__init__()

        self.dt = dt
        self.render_mode = render_mode

        # Speed limits
        self.max_speed = 40.0  # m/s (~144 km/h)
        self.min_speed = 0.0

        # Acceleration limits
        self.max_accel = 3.0   # m/s^2
        self.min_accel = -6.0  # m/s^2

        # Headway / safety
        self.desired_time_headway = 1.8  # seconds
        self.min_gap = 2.0               # minimum distance [m]

        # Observation space: [ego_speed, lead_speed, distance_gap, relative_speed]
        high = np.array(
            [self.max_speed, self.max_speed, 200.0, self.max_speed],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # --- DISCRETE ACTIONS FOR DQN ---
        # 5 discrete acceleration commands
        self.accel_values = np.array(
            [self.min_accel, -3.0, 0.0, 1.5, self.max_accel],
            dtype=np.float32
        )
        self.n_actions = len(self.accel_values)
        self.action_space = spaces.Discrete(self.n_actions)

        # Internal state variables
        self.ego_speed = None
        self.lead_speed = None
        self.distance_gap = None
        self.t = 0.0

    def _get_obs(self):
        rel_speed = self.lead_speed - self.ego_speed
        return np.array(
            [self.ego_speed, self.lead_speed, self.distance_gap, rel_speed],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use Gymnasium's internal RNG self.np_random
        self.ego_speed = float(self.np_random.uniform(15.0, 25.0))   # m/s
        self.lead_speed = float(self.np_random.uniform(15.0, 25.0))  # m/s
        self.distance_gap = float(self.np_random.uniform(20.0, 40.0))  # m

        self.t = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Convert action (can be scalar, list, or array) to a single int index
        action_idx = int(np.array(action).reshape(-1)[0])
        action_idx = int(np.clip(action_idx, 0, self.n_actions - 1))
        accel = float(self.accel_values[action_idx])

        # Ego vehicle dynamics
        self.ego_speed = float(
            np.clip(
                self.ego_speed + accel * self.dt,
                self.min_speed,
                self.max_speed
            )
        )

        # Lead car random accel
        lead_accel = float(self.np_random.uniform(-1.0, 1.0))
        self.lead_speed = float(
            np.clip(
                self.lead_speed + lead_accel * self.dt,
                self.min_speed,
                self.max_speed
            )
        )

        # Distance gap update
        self.distance_gap += (self.lead_speed - self.ego_speed) * self.dt
        self.distance_gap = max(self.distance_gap, 0.0)

        self.t += self.dt

        # --- Reward calculation ---
        collision = self.distance_gap <= self.min_gap
        desired_gap = self.desired_time_headway * self.ego_speed
        gap_error = self.distance_gap - desired_gap

        r_gap = -0.01 * (gap_error ** 2)
        r_comfort = -0.001 * (accel ** 2)
        r_speed = -0.1 if self.ego_speed > 35.0 else 0.0

        reward = r_gap + r_comfort + r_speed

        done = False
        if collision:
            reward -= 50.0
            done = True

        if self.t >= 60.0:
            done = True

        obs = self._get_obs()
        info = {"collision": collision}
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), done, truncated, info

    def render(self):
        print(
            f"t={self.t:5.1f}s | "
            f"ego_v={self.ego_speed:5.2f} m/s | "
            f"lead_v={self.lead_speed:5.2f} m/s | "
            f"gap={self.distance_gap:6.2f} m"
        )

    def close(self):
        pass

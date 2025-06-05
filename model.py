from prophet import Prophet
import pandas as pd
import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np

class CpuSchedulingEnv(gym.Env):
    def __init__(self, cpu_predictions_list, max_tasks=5, max_tasks_per_hour=1):
        super().__init__()
        self.cpu_predictions_list = cpu_predictions_list  # list of 24-length np arrays
        self.max_tasks = max_tasks
        self.max_tasks_per_hour = max_tasks_per_hour
        self.current_task = 0
        self.predicted_cpu_usage = None
        self.task_priorities = None
        self.task_durations = None
        self.scheduled = None

        self.action_space = spaces.Discrete(24)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24 + 24 + 2,), dtype=np.float32)
        self.episode_index = 0

    def reset(self):
        self.current_task = 0
        self.scheduled = np.zeros(24)

        # Make sure these match max_tasks exactly
        self.task_priorities = np.random.rand(self.max_tasks)
        self.task_durations = np.random.randint(1, 4, size=self.max_tasks)

        # Handle cycling through CPU prediction dataset
        self.predicted_cpu_usage = self.cpu_predictions_list[self.episode_index % len(self.cpu_predictions_list)]
        self.episode_index += 1

        return self._get_obs()

    def _get_obs(self):
        normalized_scheduled = self.scheduled / self.max_tasks_per_hour

        if self.current_task < self.max_tasks:
            current_priority = np.array([self.task_priorities[self.current_task]])
            current_duration = np.array([self.task_durations[self.current_task] / 3.0])
        else:
            # Dummy values to avoid index errors after all tasks are scheduled
            current_priority = np.array([0.0])
            current_duration = np.array([0.0])

        obs = np.concatenate([self.predicted_cpu_usage, normalized_scheduled, current_priority, current_duration])
        return obs.astype(np.float32)

    def _can_schedule(self, start_hour, duration):
        # Check if scheduling this task over [start_hour, start_hour+duration) exceeds max load or day length
        if start_hour + duration > 24:
            return False
        return all(self.scheduled[h] < self.max_tasks_per_hour for h in range(start_hour, start_hour + duration))

    def step(self, action):
        done = False
        reward = 0

        duration = self.task_durations[self.current_task]

        if not self._can_schedule(action, duration):
            reward = -2.0  # Penalty for invalid scheduling
        else:
            # Reward low CPU usage over all scheduled hours + priority bonus
            cpu_load = np.mean(self.predicted_cpu_usage[action:action+duration])
            priority = self.task_priorities[self.current_task]
            reward = -cpu_load + priority * 0.5

            # Schedule the task
            for h in range(action, action + duration):
                self.scheduled[h] += 1

        self.current_task += 1
        if self.current_task >= self.max_tasks:
            done = True

        return self._get_obs(), reward, done, {}


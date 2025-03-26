import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make('PandaReach-v3', render_mode="human")
env = Monitor(env, "logs/") 

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_panda_tensorboard/")
model.learn(total_timesteps=100000)

# Save model
model.save("panda_rl_model")

# import gymnasium as gym
# import panda_gym
# import torch
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# import os
# import matplotlib.pyplot as plt
# import pandas as pd

# # Create environment and enable logging
# log_dir = "logs/"
# os.makedirs(log_dir, exist_ok=True)

# env = gym.make('PandaReach-v3', render_mode="rgb_array")
# env = Monitor(env, log_dir)  # Saves logs for analysis

# # Define the PPO model with PyTorch backend
# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_panda_tensorboard/")

# # Train the model
# model.learn(total_timesteps=10000)

# # Save the trained model
# model.save("panda_rl_model")
# env.close()

# print("Training completed and model saved.")

# # Load training logs
# monitor_file = os.path.join(log_dir, [f for f in os.listdir(log_dir) if "monitor" in f][0])
# data = pd.read_csv(monitor_file, skiprows=1)

# # Plot Reward Curve
# plt.figure(figsize=(10, 5))
# plt.plot(data["r"], label="Episode Reward", color="blue", alpha=0.6)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Training Progress of PPO on PandaReach")
# plt.legend()
# plt.grid()
# plt.show()

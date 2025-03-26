import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import time

# Custom Grid Environment
class GridEnv(gym.Env):
    def __init__(self, size=5):
        super(GridEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.obstacles = self.generate_obstacles()
        self.reset()
        self.fig, self.ax = plt.subplots()
        self.render_grid()

    def generate_obstacles(self):
        obstacles = set()
        for _ in range(self.size // 2):  # Add obstacles to the grid
            obstacles.add((random.randint(1, self.size-2), random.randint(1, self.size-2)))
        return obstacles

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size-1, self.size-1])
        return self.agent_pos

    def step(self, action):
        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}  # Up, Down, Left, Right
        new_pos = tuple(self.agent_pos + moves[action])
        
        # Check boundaries and obstacles
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and new_pos not in self.obstacles):
            self.agent_pos = np.array(new_pos)
        
        reward = -0.1  # Small step penalty
        done = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10  # Reward for reaching the goal
            done = True
        
        return self.agent_pos, reward, done, {}
    
    def render_grid(self):
        self.ax.clear()
        grid = np.zeros((self.size, self.size))
        for obs in self.obstacles:
            grid[obs] = -1  # Mark obstacles
        grid[self.agent_pos[0], self.agent_pos[1]] = 1  # Agent
        grid[self.goal_pos[0], self.goal_pos[1]] = 2  # Goal
        
        self.ax.imshow(grid, cmap='coolwarm', origin='upper')
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True, color='black', linewidth=1)
        plt.pause(0.1)

    def render(self):
        self.render_grid()

# DQN Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.gamma = 0.9
        self.epsilon = 0.2
    
    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),  # Explicit input layer
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.env.action_space.n, activation='linear')
        ])
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])
    
    def train(self, episodes=1000, save_path='dqn_model.h5'):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                target_q = self.model.predict(np.array([state]), verbose=0)
                target_q[0][action] = reward + self.gamma * np.max(self.target_model.predict(np.array([next_state]), verbose=0))
                self.model.fit(np.array([state]), target_q, epochs=1, verbose=0)
                state = next_state
                total_reward += reward
                self.env.render()  # Render grid during training
            self.target_model.set_weights(self.model.get_weights())
            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")
        self.model.save(save_path)
        plt.plot(rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Training Performance')
        plt.show()

    def test(self, episodes=10, load_path='dqn_model.h5'):
        self.model = keras.models.load_model(load_path)
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            print(f"Test Episode {episode + 1}")
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(np.array([state]), verbose=0))
                state, _, done, _ = self.env.step(action)
                time.sleep(0.1)  # Add delay for better visualization

# Train and test DQN
env = GridEnv()
dqn_agent = DQNAgent(env)
dqn_agent.train(1000)
dqn_agent.test(10)
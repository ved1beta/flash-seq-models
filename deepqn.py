import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import gymnasium as gym

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr

        self.q_network = DeepQNetwork(state_size, action_size)
        self.target_network = DeepQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
    
        return np.argmax(q_values.cpu().numpy())
    def replay(self, batch_size = 32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        


def train_dqn(episodes=500):
    """Train DQN agent and return training metrics"""

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium returns (observation, info)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        scores_window.append(total_reward)
        scores.append(total_reward)

        if len(agent.memory) > 32:
            agent.replay(32)
            

        if episode % 100 == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')
            
    env.close()
    return scores, agent

def visualize_training(scores):
    """Create visualizations of training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episode scores
    ax1.plot(scores)
    ax1.set_title('Episode Scores Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # Plot 2: Moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(scores)), moving_avg)
        ax2.set_title(f'Moving Average (window={window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Score')
        ax2.grid(True)
    
    # Plot 3: Score distribution
    ax3.hist(scores, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_title('Score Distribution')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    # Plot 4: Learning curve (last 100 episodes)
    last_episodes = min(100, len(scores))
    recent_scores = scores[-last_episodes:]
    ax4.plot(range(len(scores)-last_episodes+1, len(scores)+1), recent_scores)
    ax4.set_title(f'Last {last_episodes} Episodes')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Total Episodes: {len(scores)}")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Best Score: {np.max(scores):.2f}")
    print(f"Final 100 Episode Average: {np.mean(scores[-100:]):.2f}")

def test_agent(agent, episodes=10, render=False):
    """Test the trained agent"""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium returns (observation, info)
        
        total_reward = 0
        done = False
        
        while not done:
            # Use trained policy (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
        test_scores.append(total_reward)
        print(f'Test Episode {episode + 1}: Score = {total_reward}')
    
    env.close()
    print(f'\nTest Results - Average Score: {np.mean(test_scores):.2f}')
    return test_scores

if __name__ == "__main__":
    print("Starting DQN Training...")
    print("Environment: CartPole-v1")
    print("=" * 50)
    
    # Train the agent
    scores, trained_agent = train_dqn(episodes=500)
    
    # Visualize results
    visualize_training(scores)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_scores = test_agent(trained_agent, episodes=5)
    
    print("\nTraining completed!")

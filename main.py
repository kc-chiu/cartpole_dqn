import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Number of games to be played by the agent
EPISODES = 500

class Agent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0] # Number of properties of environment state
        self.action_size = env.action_space.n # Number of agent's possible actions
        self.memory = deque(maxlen=2000) # records of state, action, reward etc after each action
        self.gamma = 0.95    # discount rate for computing future discounted reward
        self.exploration_rate = 1.0  # randomness of agent's action decision, 1 means totally random.
        self.exploration_rate_min = 0.01 # minimum randomness of agent's action decision
        self.exploration_rate_decay = 0.995 # randomnes of agent's action decision decays over time
        self.learning_rate = 0.001 # learning rate of the deep Q-learning model
        self.model = self._q_model() # the deep Q-learning model

    def _q_model(self):
        # Neural Net for Deep Q-learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    # Initialize gym environment and agent
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    
    # Set pre_trained to False to train the agent
    # Set pre_trained to True to use pre-trained model
    # make sure the path to the pre-trained model is correct
    # cartpole-dqn-500.h5 is the model that can achieve score of 499 in about 80% of games
    pre_trained = False
    if pre_trained:
        agent.load('./model/cartpole-dqn-500.h5')
        agent.exploration_rate = 0.0
    
    # Batch Size of experience tuples samples from memory for training
    BATCH_SIZE = 32
    # Aim at surviving the game as long as this time
    TIME_GOAL = 500
    # Record the score of each game  
    scores = []

    # Iterate the game
    for e in range(EPISODES):
        
        # Reset state at the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        
        # time represents how long each game lasts
        # the more time the more score
        # the goal is keep the pole straight as long as possible until score of time goal
        for time in range(TIME_GOAL):
            
            # Display cartpole movement when using pre-trained model
            if pre_trained:
                env.render()
            
            # Decide on action
            action = agent.act(state)

            # Feedback on action
            # reward is 1 each time the pole survived
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            # Advance the game to the next state
            next_state = np.reshape(next_state, [1, agent.state_size])
            # Remember the last state, action, etc. for training
            agent.remember(state, action, reward, next_state, done)
            # Make the next state the new current state
            state = next_state
            
            # done becomes True if game ends
            # print the result then break to start a new episode
            if done:
                print('episode: {}/{}, score: {}, exploration rate: {:.2}'
                      .format(e, EPISODES, time, agent.exploration_rate))
                scores.append(time)
                break
            
            # Train the agent with the experience gained in the episode
            if not pre_trained and len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
        
        if e % 10 == 0 or time >= TIME_GOAL-1:
            agent.save('./model/cartpole-dqn.h5')
        
        if not pre_trained and time >= TIME_GOAL-1:
            break

    # Plot the scores of each game episode
    plt_title = 'using Pre-trained Model' if pre_trained else 'during Training'
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('CartPole Game Score {}'.format(plt_title))
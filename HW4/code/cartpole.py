import numpy as np
import gym
import os
import random
from tqdm import tqdm

total_reward = []
episode = 3000
decay = 0.045


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.5, GAMMA=0.97, num_bins=7):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: The discount factor (tradeoff between immediate rewards and future rewards)
            num_bins: Number of part that the continuous space is to be sliced into.
        """
        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        self.qtable = np.zeros((self.num_bins, self.num_bins,
                               self.num_bins, self.num_bins, self.env.action_space.n))

        # init_bins() is your work to implement.
        self.bins = [
            self.init_bins(-2.4, 2.4, self.num_bins),  # cart position
            self.init_bins(-3.0, 3.0, self.num_bins),  # cart velocity
            self.init_bins(-0.5, 0.5, self.num_bins),  # pole angle
            self.init_bins(-2.0, 2.0, self.num_bins)  # tip velocity
        ]

    def init_bins(self, lower_bound, upper_bound, num_bins):
        """
        Slice the interval into #num_bins parts.
        Parameters:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            num_bins: Number of parts to be sliced.
        Returns:
            a numpy array of #num_bins - 1 quantiles.
        """
        # Begin your code
        # Compute the quantiles and return them as an array
        return np.linspace(lower_bound, upper_bound, num_bins, endpoint=False)[1:]
        # Since the first and last quantile are the lower and upper bound,
        # we only need to return the quantiles from the second to the second last.
        # End your code

    def discretize_value(self, value, bins):
        """
        Discretize the value with given bins.
        Parameters:
            value: The value to be discretized.
            bins: A numpy array of quantiles
        returns:
            The discretized value.
        """
        # Begin your code
        # Find the index of the quantile that the value belongs to
        return np.digitize([value], bins)[0]
        # End your code

    def discretize_observation(self, observation):
        """
        Discretize the observation which we observed from a continuous state space.
        Parameters:
            observation: The observation to be discretized, which is a list of 4 features:
                1. cart position.
                2. cart velocity.
                3. pole angle.
                4. tip velocity.
        Returns:
            state: A list of 4 discretized features which represents the state.
        """
        # Begin your code
        state = []
        for i in range(len(observation)):
            s = self.discretize_value(observation[i], self.bins[i]) # discretize each feature in observation
            state.append(s)
        return tuple(state) # return the discretized state in tuple form
        # End your code

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.
        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.
        Returns:
            action: The action to be evaluated.
        """
        # Begin your code
        if np.random.uniform(0, 1) < self.epsilon: # Explore
            action = self.env.action_space.sample()
        else:  # Exploit
            action = np.argmax(self.qtable[state])
        return action
        # End your code

    def learn(self, state, action, reward, next_state, done):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.
        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.
        Returns:
            None (Don't need to return anything)
        """
        # Begin your code
        # 2 cases(done / not doen) to update the qtable 
        if not done:
            cur_qvalue = self.qtable[state+(action,)]
            next_max = np.max(self.qtable[next_state])
            # If the episode is not done, we need to consider both the current and future reward
            new_qvalue = (1 - self.learning_rate) * cur_qvalue + self.learning_rate * (reward + self.gamma * next_max)
            self.qtable[state+(action,)] = new_qvalue
        else: 
            cur_qvalue = self.qtable[state+(action,)]
            next_max = np.max(self.qtable[next_state])
            # If the episode is done, we only need to consider the current reward
            new_qvalue = (1 - self.learning_rate) * cur_qvalue + self.learning_rate * (reward)
            self.qtable[state+(action,)] = new_qvalue
        # End your code
        if done:
            np.save("./Tables/cartpole_table.npy", self.qtable)

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        state = self.discretize_observation(self.env.reset())
        max_q = np.max(self.qtable[state])
        return max_q
        # End your code


def train(env):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    training_agent = Agent(env)
    rewards = []
    for ep in tqdm(range(episode)):
        state = training_agent.discretize_observation(env.reset())
        done = False

        count = 0
        while True:
            count += 1
            action = training_agent.choose_action(state)
            next_observation, reward, done, _ = env.step(action)

            next_state = training_agent.discretize_observation(next_observation)
            training_agent.learn(state, action, reward, next_state, done)

            if done:
                rewards.append(count)
                break
            state = next_state

        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay
        np.save("./Tables/cartpole_table.npy", training_agent.qtable)
    total_reward.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)

    testing_agent.qtable = np.load("./Tables/cartpole_table.npy")
    rewards = []

    for _ in range(100):
        state = testing_agent.discretize_observation(testing_agent.env.reset())
        count = 0
        while True:
            count += 1
            action = np.argmax(testing_agent.qtable[tuple(state)])
            next_observation, _, done, _ = testing_agent.env.step(action)

            if done == True:
                rewards.append(count)
                break

            next_state = testing_agent.discretize_observation(next_observation)
            state = next_state

    print(f"average reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    os.makedirs("./Tables", exist_ok=True)

   # training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/cartpole_rewards.npy", np.array(total_reward))

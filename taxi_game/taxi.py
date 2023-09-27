import gym
import numpy as np
import time

EPISODE_LENGTH = 100
LR = 0.1


class Agent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_action = n_actions
        self.model = np.random.rand(n_states, n_actions)

    def select_action(self, state):
        # return np.argmax(self.model[state])
        return np.random.randint(6)

    def update_agent(self, elite_trajectories, lr=LR):
        new_model = np.zeros_like(self.model)
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.n_states):
            state_sum = np.sum(new_model[state])
            if state_sum > 0:
                new_model[state] /= state_sum
        self.model = (1 - lr) * self.model + (lr * self.model)


class Environment:
    def __init__(self, visualize=True):
        self.env = gym.make('Taxi-v3')
        self.num_states = self.env.observation_space.n  # 500 = 25 pos * 5 passenger locations * 4 destinations
        self.num_actions = self.env.action_space.n     # 6 = N, S, W, E, drop, pick up
        self.visualize = visualize

    def initial_state(self):
        return self.env.reset()

    def take_action(self, action):
        return self.env.step(action)

    def start_episode(self, agent, max_len=EPISODE_LENGTH):
        states, actions, rewards = [[]] * 3

        state = self.initial_state()
        for _ in range(max_len):

            action = agent.select_action(state)
            next_state, reward, done, _ = self.take_action(action)
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            self.draw()
            if done:
                break
            state = next_state

        trajectory = {'states': states,
                      'actions': actions,
                      'rewards': rewards}
        return trajectory

    def draw(self):
        if self.visualize:
            self.env.render()

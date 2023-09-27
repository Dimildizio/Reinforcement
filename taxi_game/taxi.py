import gym
import numpy as np
import time


EPISODE_LENGTH = 100


class Agent:
    def select_action(self, state):
        return np.random.randint(6)


class Environment:
    def __init__(self):
        self.env = gym.make('Taxi-v3')
        self.num_states = self.env.observation_space.n  # 500 = 25 pos * 5 passenger locations * 4 destinations
        self.num_actions = self.env.action_space.n     # 6 = N, S, W, E, drop, pick up

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

            if done:
                break
            state = next_state

        trajectory = {'states': states,
                      'actions': actions,
                      'rewards': rewards}
        return trajectory

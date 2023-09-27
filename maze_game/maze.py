import gym
import gym_maze
import numpy as np
import time
from abc import ABCMeta
from typing import List, Dict, Tuple
from dataclasses import dataclass


class Maze:
    def __init__(self):
        self.env = gym.make('maze-sample-5x5-v0')

    def reset(self):
        return self.env.reset()

    def step(self, action: int) -> int:
        return self.env.step(action)

    def render(self) -> None:
        self.env.render()


class Agent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def get_action(self, state: int):
        raise NotImplementedError('Only child classes should implement get_action')

    def fit(self, elite_trajectories: List):
        raise NotImplementedError('Only child classes should implement fit')


class RandomAgent(Agent, metaclass=ABCMeta):
    def get_action(self, state: int) -> int:
        return int(np.random.randint(self.n_actions))


class CrossEntropyAgent(Agent):
    def __init__(self, n_actions: int = 4, n_states: int = 25):
        super().__init__(n_actions)
        self.n_states = n_states
        self.model = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def get_action(self, state: int) -> int:
        actions = np.arange(self.n_actions)
        result = np.random.choice(actions, p=self.model[state])
        return int(result)

    @property
    def zero_matrix(self) -> np.ndarray:
        return np.zeros((self.n_states, self.n_actions))

    def fit(self, elite_trajectories: List) -> None:
        new_model = self.zero_matrix
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state, action] += 1

        for state in range(self.n_states):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model


@dataclass
class Runner:
    agent: Agent
    env: Maze
    q_param: float = 0.9
    iters_num: int = 20
    traj_num: int = 50

    def get_state(self, obs: Tuple) -> int:
        state = np.sqrt(self.agent.n_states) * obs[0] + obs[1]
        return int(state)

    @property
    def new_state(self) -> int:
        obs = self.env.reset()
        return self.get_state(obs)

    @property
    def new_trajectory(self) -> Dict:
        return {'states': [], 'actions': [], 'rewards': []}

    def get_trajectory(self, max_len: int = 1000, vis: bool = False) -> Dict:
        trajectory = self.new_trajectory
        state = self.new_state
        for _ in range(max_len):
            trajectory['states'].append(state)
            action = self.agent.get_action(state)
            trajectory['actions'].append(action)
            obs, reward, done, _ = self.env.step(action)
            trajectory['rewards'].append(reward)
            state = self.get_state(obs)

            if vis:
                self.env.render()
                time.sleep(0.05)
            if done:
                break
        return trajectory

    def run(self):
        for iteration in range(self.iters_num):
            # Policy evaluation
            trajectories = [self.get_trajectory() for _ in range(self.traj_num)]
            total_reward = [np.sum(traj['rewards']) for traj in trajectories]
            print(f'Iteration {iteration} mean reward: {np.mean(total_reward)}')

            # Policy improvement
            quant = np.quantile(total_reward, self.q_param)
            elite_trajectories = [traj for traj in trajectories if np.sum(traj['rewards']) > quant]
            self.agent.fit(elite_trajectories)

        trajectory = self.get_trajectory(max_len=100, vis=True)
        print(f"Total reward: {sum(trajectory['rewards'])}")
        print("Model:\n", self.agent.model)

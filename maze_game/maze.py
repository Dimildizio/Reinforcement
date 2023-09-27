"""
maze.py

This script contains classes to solve a maze game using a random action strategy.

Classes:
- GameSolver: Provides methods to take actions, initialize the environment, render the game, and play turns.

Usage:
- Create a GameState instance.
- Call the mainloop method to start playing the game.
"""

import gym
import gym_maze
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List


# Constants
STATES_N = 25
ACTIONS_N = 4
TRAJECTORIES_N = 10
SLEEP_T = 0.01
TURNS = 1000
EPOCHS = 10
Q_PARAM = 0.9

@dataclass
class RandomAgent:
    """
    Class to solve a maze game using a random action strategy.
    """
    env: object
    actions_n: int = ACTIONS_N

    def take_action(self, action: int):
        """
        Take an action in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple: (next_state, reward, done, _)
        """
        result = self.env.step(action)
        return result


    @property
    def random_action(self) -> int:
        """
        Generate a random action.

        Returns:
            int: A random action.
        """
        return np.random.randint(self.actions_n)

    def get_action(self, state: int) -> int:
        """
        Get the action to take based on the current state.

        Args:
            state (int): The current state.

        Returns:
            int: The action to take.
        """
        return self.random_action

    def play_turn(self, state: int):
        """
        Play a single turn of the game.

        Args:
            state (int): The current state.
        """
        action = self.get_action(state)
        next_state, reward, done, _ = self.take_action(action)
        return next_state, reward, done, action


@dataclass
class CrossEntropyAgent(RandomAgent):
    states_n: int = STATES_N
    model:np.ndarray = None


    @property
    def zero_model(self):
        return np.ones((self.states_n, self.actions_n))
    def __post_init__(self):
        super().__init__(self.env, self.actions_n)
        if self.model is None:
            self.model = self.zero_model / self.actions_n

    def get_action(self, state) -> int:
        actions = np.arange(self.actions_n)
        probability =self.model[state]
        action = int(np.random.choice(actions, p=probability))
        return action

    def fit(self, elite_trajectories):
        new_model = self.zero_model
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] +=1

        for state in range(self.states_n):
            states_sum = np.sum(new_model[state])
            if states_sum > 0:
                new_model[state] /= states_sum
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model



class Runner:
    def __init__(self, visualize=False):
        """
        Class to control the game flow and provide information about the game's progress.
        """
        self.env = gym.make('maze-sample-5x5-v0', disable_env_checker=True)
        self.actor = CrossEntropyAgent(self.env)
        self.visualize = visualize
        self.finished = False
        self.trajectory = self.init_trajectory()

    def switch_vis(self):
        self.visualize = not self.visualize

    def init_trajectory(self) -> Dict:
        trajectory =        {'states':[],
                           'actions':[],
                           'rewards':[]}
        return trajectory


    def update_trajectory(self, state: np.ndarray, action: int, reward: int) -> None:
        self.trajectory['states'].append(state)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)

    def run_epoch(self):
        self.trajectory = self.init_trajectory()
        return self.get_trajectory()

    def iter_train(self, traj: int=TRAJECTORIES_N):
        trajectories = [self.run_epoch() for _ in range(traj)]
        total_rewards = self.get_total_reward(trajectories)
        q = np.quantile(total_rewards, Q_PARAM)
        elite_trajectories = [traj for traj in trajectories if np.sum(traj['rewards']) > q]
        return elite_trajectories

    def mainloop(self, epochs=EPOCHS):
        for epoch in range(epochs):
            elite_trajectories = self.iter_train()
            self.actor.fit(elite_trajectories)


    def get_model(self):
        self.switch_vis()
        self.get_trajectory()
        return self.actor.model

    def get_total_reward(self, trajectories):
        return np.sum([np.sum(tra['rewards']) for tra in trajectories])


    def get_trajectory(self, max_len: int=TURNS) -> Dict:
        """
        Main loop to run the game.
        """
        state = self.starting_state

        for attempt in range(max_len):
            state = self.get_next_observation(state)
            self.draw()

            if self.has_finished(attempt):
                break

        #self.check_failed()
        return self.trajectory

    def draw(self) -> None:
        """
        Render the environment and pause briefly.
        """
        if self.visualize:
            self.env.render()
            time.sleep(SLEEP_T)

    @property
    def reward_result(self) -> int:
        """
        Property to get the total reward accumulated during the game.

        Returns:
            int: The total reward.
        """
        return sum(self.trajectory['rewards'])

    @staticmethod
    def get_state(observation: tuple) -> int:
        """
        Get the state corresponding to the given observation.

        Args:
            observation (tuple): The observation.

        Returns:
            int: The state.
        """
        return int(np.sqrt(STATES_N) * observation[0] + observation[1])

    @property
    def starting_state(self) -> int:
        """
        Property to get the initial state of the game.

        Returns:
            int: The initial state.
        """
        obs = self.env.reset()
        state = self.get_state(obs)
        return state

    def get_next_observation(self, state: int) -> int:
        """
        Get the next state after playing a turn.

        Args:
            state (int): The current state.

        Returns:
            int: The next state.
        """
        next_obs, reward, done, action = self.actor.play_turn(state)
        self.finished = done
        self.update_trajectory(state, action, reward)
        state = self.get_state(next_obs)
        return state

    def has_finished(self, attempt: int) -> bool:
        """
        Check if the game has finished.

        Args:
            attempt (int): The current attempt number.

        Returns:
            bool: True if the game has finished, False otherwise.
        """
        if self.finished:
            #print(f'attempt:{attempt}\nreward:{self.reward_result}')
            return True
        else:
            return False

    def check_failed(self):
        """
        Print a message if the game was not successfully completed.
        """
        if not self.finished:
            print('ПОТРАЧЕНО')
            print(f'Reward:{self.reward_result}')


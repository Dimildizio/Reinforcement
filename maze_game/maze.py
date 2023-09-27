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


# Constants
STATE_N = 25
ACTIONS_N = 4
SLEEP_T = 0.001
TURNS = 1000


@dataclass
class GameSolver:
    """
    Class to solve a maze game using a random action strategy.
    """
    env: object
    total_reward: int = 0

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

    def initial_state(self):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: The initial observation.
        """
        return self.env.reset()

    @property
    def random_action(self) -> int:
        """
        Generate a random action.

        Returns:
            int: A random action.
        """
        return np.random.randint(ACTIONS_N)

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
        self.total_reward += reward
        return next_state, reward, done, action


class Runner:
    def __init__(self):
        """
        Class to control the game flow and provide information about the game's progress.
        """
        self.env = gym.make('maze-sample-5x5-v0', disable_env_checker=True)
        self.actor = GameSolver(self.env)
        self.finished = False
        self.states = []
        self.actions = []
        self.rewards = []

    def update_trajectory(self, state: np.ndarray, action: int, reward: int) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def mainloop(self, max_len: int=TURNS) -> None:
        """
        Main loop to run the game.
        """
        state = self.starting_state

        for attempt in range(max_len):
            state = self.get_next_observation(state)
            self.draw()

            if self.has_finished(attempt):
                break

        self.check_failed()

    def draw(self) -> None:
        """
        Render the environment and pause briefly.
        """
        self.env.render()
        time.sleep(SLEEP_T)

    @property
    def reward_result(self) -> int:
        """
        Property to get the total reward accumulated during the game.

        Returns:
            int: The total reward.
        """
        return self.actor.total_reward

    @staticmethod
    def get_state(observation: tuple) -> int:
        """
        Get the state corresponding to the given observation.

        Args:
            observation (tuple): The observation.

        Returns:
            int: The state.
        """
        return int(np.sqrt(STATE_N) * observation[0] + observation[1])

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
        self.update_trajectory(next_obs, action, reward)
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
            print(f'attempt:{attempt}\nreward:{self.reward_result}')
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


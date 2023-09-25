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
import random
import time
from dataclasses import dataclass

# Constants
STATE_N = 25
ACTIONS_N = 4
SLEEP_T = 0.01
TURNS = 1000


@dataclass
class GameSolver:
    """
    Class to solve a maze game using a random action strategy.
    """
    def __init__(self):
        self.env = gym.make('maze-sample-5x5-v0', disable_env_checker=True)
        self.total_reward = 0

    def take_action(self, action):
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

    def draw(self):
        """
        Render the environment and pause briefly.
        """
        self.env.render()
        time.sleep(SLEEP_T)

    @property
    def random_action(self):
        """
        Generate a random action.

        Returns:
            int: A random action.
        """
        return np.random.randint(ACTIONS_N)

    def get_action(self, state):
        """
        Get the action to take based on the current state.

        Args:
            state (int): The current state.

        Returns:
            int: The action to take.
        """
        return self.random_action

    @staticmethod
    def get_state(observation):
        """
        Get the state corresponding to the given observation.

        Args:
            observation (tuple): The observation.

        Returns:
            int: The state.
        """
        return int(np.sqrt(STATE_N) * observation[0] + observation[1])

    def play_turn(self, state):
        """
        Play a single turn of the game.

        Args:
            state (int): The current state.
        """
        action = self.get_action(state)
        next_state, reward, done, _ = self.take_action(action)
        self.total_reward += reward
        self.draw()
        return next_state, done

    def mainloop(self):
        """
        Main loop to play the game.
        """
        obs = self.initial_state()
        state = self.get_state(obs)
        done = False
        for n in range(TURNS):

            next_obs, done = self.play_turn(state)
            state = self.get_state(next_obs)

            if done:
                print(f'attempt:{n}\nreward:{self.total_reward}')
                break
        if not done:
            print('ПОТРАЧЕНО')
            print(f'Reward:{self.total_reward}')

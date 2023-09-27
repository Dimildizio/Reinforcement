import gym
import numpy as np

N_EPISODES = 200
TEST_EPOCHS = 10
TEST_ACTION_LIMIT = 200
ELITE_FRAC = 0.05
EPISODE_LENGTH = 100
LR = 0.1


class Agent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_action = n_actions
        self.model = np.random.rand(n_states, n_actions)

    def select_action(self, state):
        return np.argmax(self.model[state])

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
    def __init__(self, visualize=False):
        self.env = gym.make('Taxi-v3')
        self.num_states = self.env.observation_space.n  # 500 = 25 pos * 5 passenger locations * 4 destinations
        self.num_actions = self.env.action_space.n     # 6 = N, S, W, E, drop, pick up
        self.visualize = visualize

    def initial_state(self):
        return self.env.reset()

    def take_action(self, action):
        return self.env.step(action)

    def start_episode(self, agent, max_len=EPISODE_LENGTH):
        states, actions, rewards = [], [], []

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


class Trainer:
    def __init__(self, episodes=N_EPISODES, elite_fraction=ELITE_FRAC, lr=LR):
        self.episodes = episodes
        self.fraction = elite_fraction
        self.top_traj_num = int(self.episodes * self.fraction)
        self.lr = lr

    def train(self, agent, env):
        for n in range(self.episodes):
            eps = [env.start_episode(agent) for _ in range(self.episodes)]
            total_reward = [sum(episode['rewards']) for episode in eps]
            top_idxs = np.argsort(total_reward)[-self.top_traj_num:]
            elite_traj = [eps[n] for n in top_idxs]
            agent.update_agent(elite_traj, self.lr)
            print(f'Train episode {n} reward: {np.mean(total_reward)}')
        print('Finished training\n')

    def test(self, agent, env, test_episodes=TEST_EPOCHS, action_limit=TEST_ACTION_LIMIT):
        total_reward = []
        for n in range(test_episodes):
            print('TEST EPOCH:', n)
            state = env.initial_state()
            episode_reward = 0
            for n in range(action_limit):
                action = agent.select_action(state)
                state, reward, done, _ = env.take_action(action)
                episode_reward += reward
                if done:
                    total_reward.append(episode_reward)
                    break
            print(f'episode reward: {episode_reward}\n')
        avg_reward = np.mean(total_reward)
        return avg_reward

    def fit(self, agent, env):
        self.train(agent, env)
        avg_reward = self.test(agent, env)
        print('Average reward:', avg_reward)

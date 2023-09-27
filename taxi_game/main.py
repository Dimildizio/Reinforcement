from taxi import Environment, Agent


if __name__ == '__main__':
    taxi_env = Environment()
    agent = Agent(taxi_env.num_states, taxi_env.num_actions)
    taxi_env.start_episode(agent)

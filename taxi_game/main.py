from taxi import Environment, Agent, Trainer


if __name__ == '__main__':
    taxi_env = Environment()
    agent = Agent(taxi_env.num_states, taxi_env.num_actions)
    trainer = Trainer()
    trainer.fit(agent, taxi_env)

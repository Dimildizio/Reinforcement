from maze import Maze, CrossEntropyAgent, Runner


if __name__ == '__main__':
    env = Maze()
    agent = CrossEntropyAgent()
    runner = Runner(agent, env)
    runner.run()


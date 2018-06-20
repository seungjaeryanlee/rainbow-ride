#!/usr/bin/env python3
import gym

from dqn import DQN
from agent import NaiveDQNAgent


def main():
    env = gym.make('CartPole-v0')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    agent = NaiveDQNAgent(env, dqn)

    agent.train(n_steps=20000, show=True)
    total_reward = agent.play(show=True)
    print(total_reward)


if __name__ == '__main__':
    main()

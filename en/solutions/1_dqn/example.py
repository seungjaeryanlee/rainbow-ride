#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.optim as optim

from dqn import DQN
from agent import DQNAgent
from replay import ReplayBuffer

def get_epsilon_schedule(start, final, decay):
    return lambda step: final + (start - final) * np.exp(-1. * step / decay)

def main():
    USE_CUDA = torch.cuda.is_available()

    env = gym.make('CartPole-v0')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    if USE_CUDA:
        dqn = dqn.cuda()
    agent = DQNAgent(env, dqn, 
                     Optimizer=optim.Adam,
                     epsilon_schedule=get_epsilon_schedule(1.0, 0.01, 500),
                     replay_buffer=ReplayBuffer(1000))

    agent.train(n_steps=20000)
    total_reward = agent.play(render=True)
    print(total_reward)


if __name__ == '__main__':
    main()

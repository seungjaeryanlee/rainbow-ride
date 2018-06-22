#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.optim as optim

from agent import DQNAgent
from dqn import DQN
from replay import ReplayBuffer


def get_epsilon_schedule(start, end, endt, learn_start):
    """
    Generate a decreasing function that returns epsilon for each timestep.

    Parameters
    ----------
    start : float
        The epsilon value for step 0.
    end : float
        The epsilon value for step n.
    endt : int
        The number of timesteps to linearly anneal the epsilon.
    learn_start : int
        The step that the learning begins. Should be equal to DQNAgent's
        learn_start.

    Returns
    -------
    object
        A function that returns epsilon for each timestep.
    """
    return lambda step: end + max(0, (start - end) *
                                  (endt - max(0, step - learn_start)) / endt)

def main():
    USE_CUDA = torch.cuda.is_available()

    env = gym.make('CartPole-v0')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n)
    if USE_CUDA:
        dqn = dqn.cuda()
    optimizer        = optim.RMSprop(dqn.parameters(), 
                                    lr=0.00025,
                                    momentum=0.95,
                                    alpha=0.95,
                                    eps=0.01)
    epsilon_schedule = get_epsilon_schedule(start=1.0,
                                            end=0.01,
                                            endt=1000,
                                            learn_start=50)
    replay_buffer    = ReplayBuffer(capacity=1000)
    agent = DQNAgent(env, dqn, optimizer, epsilon_schedule, replay_buffer,
                    discount_factor=0.99,
                    target_update_rate=10,
                    batch_size=32,
                    learn_start=50)

    agent.train(5000)
    total_reward = agent.play(render=True)
    agent.env.close()
    print('Total Reward: ', total_reward)


if __name__ == '__main__':
    main()

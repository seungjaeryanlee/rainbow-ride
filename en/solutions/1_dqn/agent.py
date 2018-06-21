import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class NaiveDQNAgent:
    """
    A reinforcement learning agent that uses DQN naively to estimate action
    values.
    """
    def __init__(self, env, dqn, discount_factor=0.99):
        """
        Parameters
        ----------
        env
            An OpenAI Gym environment.
        dqn
            An instance of Deep Q Network written in PyTorch, inheriting
            nn.Module.
        discount_factor : float
            A float between 0 and 1 denoting the discount factor. A discount
            factor of 0 indicates a myopic agent, and a discount factor of 1
            indicates an agent that accounts all future rewards without
            discount.
        """
        self.env = env
        self.dqn = dqn
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(self.dqn.parameters())

    def act(self, state, epsilon):
        """
        Return an action with respect to the epsilon-greedy policy.

        Parameters
        ----------
        state : list of float
            The current state given by the environment.
        epsilon : float
            The possibility of selecting a random action.

        Returns
        -------
        action : int
            The action chosen by the agent.
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state)
            q_values = self.dqn.forward(state)
            action = int(q_values.argmax())
        else:
            action = self.env.action_space.sample()

        return action

    def train(self, n_steps=1000, show=False):
        """
        Train the agent for specified number of steps.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to train the agent for.
        show : bool
            If True, plots the episode rewards and losses on Jupyter Notebook.
        """
        epsilon = 0.01
        state = self.env.reset()
        done = False
        episode_rewards = []
        episode_reward = 0
        losses = []
        for step_i in range(n_steps):
            action = self.act(state, epsilon)
            next_state, reward, done, _  = self.env.step(action)
            episode_reward += reward

            self.optimizer.zero_grad()
            loss = self._compute_loss(state, action, reward, next_state, done)
            losses.append(loss)
            loss.backward()
            self.optimizer.step()

            if done:
                state = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
            else:
                state = next_state
                
        if show:
            self._plot(step_i, episode_rewards, losses)

    def _compute_loss(self, state, action, reward, next_state, done):
        """
        Compute the MSE loss given a transition (s, a, r, s').
        
        Parameters
        ----------
        state : list of float
            The current state s given by the environment.
        action : int
            Action a chosen by the agent.
        reward : float
            Reward given by the environment for given state s and action a.
        next_state : list of float
            The resulting state s' after taking action a on state s.
        done: bool
            True if s' is a terminal state. False otherwise.

        Returns
        -------
        loss : torch.tensor
            The MSE loss of the DQN.
        """
        state = torch.FloatTensor(state)
        q_values = self.dqn(state)
        q_value = q_values[action]

        next_state = torch.FloatTensor(next_state)
        next_q_values = self.dqn(next_state)
        next_q_value = next_q_values.max()

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * next_q_value

        loss = (q_value - target).pow(2).mean()

        return loss

    def _plot(self, step, rewards, losses):
        """
        Plot the total episode rewards and losses per timestep.

        Parameters
        ----------
        rewards : list of float
            List of total rewards for each episode.
        losses : list of float
            List of losses for each timestep.
        """
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('Total Episode Reward')
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('MSE Loss')
        plt.plot(losses)
        plt.show()

    def play(self, show=True):
        """
        Play an episode and return the total reward for the episode.

        Parameters
        ----------
        show : bool
            If true, render the environment.
        """
        total_reward = 0
        state = self.env.reset()
        done = False

        while not done:
            action = self.act(state, epsilon=0)
            next_state, reward, done, _  = self.env.step(action)
            if show:
                self.env.render()
            total_reward += reward
            state = next_state
        self.env.close()

        return total_reward

class DQNAgent:
    """
    A reinforcement learning agent that uses DQN specified by DeepMind's 2015
    paper to estimate action values.
    """
    def __init__(self, env, dqn, Optimizer,
                 epsilon_schedule,
                 replay_buffer,
                 discount_factor=0.99,
                 target_update_rate=64,
                 batch_size=32,
                 min_buffer_size=100):
        self.env = env
        self.dqn = dqn
        self.target_dqn = copy.deepcopy(dqn)
        self.optimizer = Optimizer(dqn.parameters())
        self.epsilon_schedule = epsilon_schedule
        self.replay_buffer = replay_buffer
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size

    def act(self, state, epsilon):
        """
        Return an action with respect to the epsilon-greedy policy.

        Parameters
        ----------
        state : list of float
            The current state given by the environment.
        epsilon : float
            The possibility of selecting a random action.

        Returns
        -------
        action : int
            The action chosen by the agent.
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state)
            q_values = self.dqn(state)
            action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()
        return action

    def _compute_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action     = torch.LongTensor(action)
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(done)

        q_values = self.dqn(state)
        q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_dqn(next_state)
        next_q_value  = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount_factor * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        return loss

    def _update_parameters(self, loss):
        """
        Update parameters with the given loss.

        Parameters
        ----------
        loss
            The temporal difference loss of Q Learning.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _plot(self, frame_idx, rewards, losses, epsilons):
        """
        Plot the total episode rewards and losses per timestep.

        Parameters
        ----------
        rewards : list of float
            List of total rewards for each episode.
        losses : list of float
            List of losses for each timestep.
        epsilons : list of float
            List of epsilons for each timestep.
        """
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('Episodic Reward')
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('Loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('Epsilon')
        plt.plot(epsilons)
        plt.tight_layout()
        plt.show()

    def train(self, n_steps=10000):
        """
        Train the agent for specified number of steps.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to train the agent for.
        """
        all_rewards = []
        losses = []
        epsilons = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, n_steps + 1):
            if frame_idx % self.target_update_rate == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            epsilon = self.epsilon_schedule(frame_idx)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.append(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > self.min_buffer_size:
                loss = self._compute_loss()
                self._update_parameters(loss)
                losses.append(loss.item())
                epsilons.append(epsilon)

        self._plot(frame_idx, all_rewards, losses, epsilons)

    def play(self, render=True):
        """
        Play an episode and return the total reward for the episode.

        Parameters
        ----------
        render : bool
            If true, render the environment.
        """
        done = False
        state = self.env.reset()
        total_reward = 0
        while not done:
            action = self.act(state, epsilon=0)
            next_state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            total_reward += reward
            state = next_state

        return total_reward

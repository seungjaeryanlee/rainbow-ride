import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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

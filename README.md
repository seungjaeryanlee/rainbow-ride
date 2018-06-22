# rainbow-ride

Explaining Rainbow DQN with PyTorch

## What is Rainbow Ride?

Rainbow Ride is a collection of PyTorch implementations of DQN papers (DQN, DDQN, PER, Dueling, Distributional, NoisyNet, Rainbow). Each implementation contains both Python scripts (`.py`) and Jupyter Notebooks (`.ipynb`). The Jupyter notebook explains the code in the Python script files for those unfamiliar with PyTorch or DQN.

To accommodate those without GPU, we use the simplest possible environment: **CartPole**. Those who have a GPU might want to use **Pong** or other Atari 2600 game environments for a better comparison with the results shown in the DQN papers.

## Installation via Conda

First, install packages using the `environment.yml` file in root.

```
conda create env -f environment.yml
```

Then, find the appropriate version of `pytorch` and `torchvision` from [pytorch.org](https://pytorch.org/) and install it.

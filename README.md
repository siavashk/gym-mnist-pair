# gym-mnist-pair
OpenAI Gym Enviroment for Matching MNIST Pairs

## Introduction
The goal of this gym environment is to provide a minimal setting for reinforcement learning in the context of image matching.

The environment consists of a pair of mnist digits against a zero background. The dimension of each image is 40-by-40, mnist digits are standard 28-by-28 pixels. The location of each mnist digit is uniformly random and independent.

In addition to pixel data, the environment also implements a pair of 28-by-28 windows. Each window is overlayed on one of the images. The state of the agent in the environment is simply the location of the upper left corner of these windows. Since there are 40-28+1=13 places where the corner can be along each dimension, the environment spans 169 distinct states.

At each step, the agent can perform the following actions for each window: "do nothing", or "move {right, up, left, down}". Given that there are two windows, the agent can perform a total of 5*5 = 25 different actions at each step.

## Dependencies
This package requires [gym](https://github.com/openai/gym) framework to be installed beforehand.

You will also need to add the mnist training dataset path to your environment.

```bash
    $ export MNIST_TRAIN_PATH='/path/to/train-images-idx3-ubyte'
```

If you clone this repository, you can find this dataset under `gym_mnist_pair/envs/train-images-idx3-ubyte`.

## Installation
Clone the repository and install with pip:

```bash
$ git clone https://github.com/siavashk/gym-mnist-pair.git
$ cd gym-mnist-pair
$ pip install -e .
```

## Usage
You can use this environment similar to any other gym environment:

```python
>>> import gym
>>> import gym_mnist_pair
>>> env = gym.make('mnist-pair-v0')
>>> env.reset() # Creates an image pair from a random mnist digit
>>> env.step(1) # The agent takes a step. Valid steps fall within [0,24]
>>> env.render() # Shows the current state of the environment
```

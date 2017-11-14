from gym.envs.registration import register

register(
    id='mnist-pair-v0',
    entry_point='gym_mnist_pair.envs:MnistPairEnv',
)

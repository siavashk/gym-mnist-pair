import gym
from gym import error, spaces, utils
from gym.utils import seeding
import struct
from array import array
import numpy as np

def state_transition_from_direction(d):
    # Five actions for each block: nothing(0), right(1), up(2), left(3), down(4):
    if type(d) != int || d < 0 || d > 4:
        raise ValueError('Unsupported action, got {}'.format(d))
        return (0, 0)
    else:
        if d == 0:
            return (0, 0)
        elif d == 1:
            return (1, 0)
        elif d == 2:
            return (0, 1)
        elif d == 3:
            return (-1, 0)
        else:
            return (-1, -1)

def get_transition(action):
'''
    Five actions for each block: nothing(0), right(1), up(2), left(3), down(4):
    For pair of blocks the following table of actions is available:
    00 | 0    10 | 5    20 | 10    30 | 15    40 | 20
    01 | 1    11 | 6    21 | 11    31 | 16    41 | 21
    02 | 2    12 | 7    22 | 12    32 | 17    42 | 22
    03 | 3    13 | 8    23 | 13    33 | 18    43 | 23
    04 | 4    14 | 9    24 | 14    34 | 19    44 | 24
'''
    if type(action) != int || action < 0 || action > 24:
        raise ValueError('Unsupported action, got {}'.format(action))
        return (0, 0, 0, 0)
    else:
        st1 = state_transition_from_direction(a / 5)
        st2 = state_transition_from_direction(a % 5)
        return np.array([st1[0], st1[1], st2[0], st2[1]])

def load_mnist():
    with open('train-images-idx3-ubyte', 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
        return np.array(images)

def makeTransMnist(image, inLength, outLength):
    image = image.reshape(inLength, inLength)
    topLeft = random_integers(low=0, high=outLength - inLength, size=2)
    transNist = np.zeros((outLength, outLength))
    transNist[topLeft[0]:topLeft[0]+inLength, topLeft[1]:topLeft[1]+inLength] = image
    return transNist

def render_image(image, block, length, threshold):
    render = ''
    for i in range(image.shape[0])):
        for j in range(image.shape[1])):
            if image[i, j] > threshold:
                if i >= block[0] && block[0] + length > i && j >= block[1] && block[1] + length > j:
                    render += "\x1b[1;%dm" % (30+1) + '@' + "\x1b[0m"
                else:
                    render += '@'
            else:
                if i >= block[0] && block[0] + length > i && j >= block[1] && block[1] + length > j:
                    render += "\x1b[1;%dm" % (30+1) + '.' + "\x1b[0m"
                else:
                    render += '.'
        render += '\n'
        return render

class MnistPairEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.mnist = load_mnist()
        self.in_image_length = int(np.sqrt(self.mnist.shape[1]))
        self.number_of_images = self.mnist.shape[0]
        self.out_image_length = 40
        self.state = np.array([0, 0, 0, 0]) # format (top_left_1_x, top_left_1_y, top_left_2_x, top_left_2_y)
        self.total_episode_steps = 100
        self.current_step = 0
        self.pair1 = np.zeros((self.out_image_length, self.out_image_length))
        self.pair2 = np.zeros((self.out_image_length, self.out_image_length))

    def _step(self, action):
        self.state += get_transition(action)
        self.current_step += 1
        np.clip(self.state, 0, self.out_image_length - self.in_image_length)

        crop1 = image_pair_1[self.state[0]:self.state[0]+self.in_image_length, self.state[1]:self.state[0]+self.in_image_length]
        crop2 = image_pair_2[self.state[2]:self.state[2]+self.in_image_length, self.state[3]:self.state[3]+self.in_image_length]

        reward = np.corrcoef(crop1.ravel(), crop2.ravel())[0, 1]
        return self.state, reward, self.current_step >= self.total_episode_steps, {}

    def _reset(self):
        self.state = np.array([0, 0, 0, 0])
        idx = np.random.randint(0, self.number_of_images)
        self.pair1 = makeTransMnist(self.mnist[idx, :])
        self.pair2 = makeTransMnist(self.mnist[idx, :])
        self.current_step = 0

    def _render(self, mode='human', close=False):
        render_image(self.pair1, self.state[0:2], self.in_image_length, 0)
        render_image(self.pair2, self.state[2:4], self.in_image_length, 0)

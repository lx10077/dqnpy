import random
import numpy as np
from collections import deque, defaultdict
from tensorboardX import SummaryWriter


class Memory(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        self.mem.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))


class ResultsBuffer(object):
    def __init__(self, base_path):
        self.buffer = defaultdict(list)
        self.index = defaultdict(int)
        self.summary_writer = SummaryWriter(base_path)

    def update_info(self, info):
        for key in info:
            self.summary_writer.add_scalar(key, info[key], self.index[key])
            self.index[key] += 1

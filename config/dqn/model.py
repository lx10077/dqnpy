from util import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Q_Net(nn.Module):
    def __init__(self, action_n):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(64 * 7 * 7, 512)
        self.fc5 = nn.Linear(512, action_n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DQN:
    def __init__(self, n_ac, lr=1e-4, discount=0.99):
        self.net = Q_Net(n_ac)
        self.target_net = Q_Net(n_ac)
        self.discount = discount
        self.n_ac = n_ac

        if USE_CUDA:
            self.net.cuda()
            self.target_net.cuda()

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def update(self, states, actions, rewards, next_states, dones, double=True):
        batch_size = states.shape[0]

        q_values_next_target = to_numpy(self.target_net(to_tensor(next_states)))
        discount_factors = self.discount * np.invert(dones).astype(np.float32)

        if double:
            best_actions = np.argmax(to_numpy(self.net(to_tensor(next_states))), axis=1)
        else:
            best_actions = np.argmax(q_values_next_target, axis=1)
        targets = rewards + discount_factors * q_values_next_target[np.arange(batch_size), best_actions]
        targets = to_tensor(targets).view(-1, 1)

        predictions = self.net(to_tensor(states)).gather(1, to_tensor(actions, dtype="LONG").view(-1, 1))

        max_q_value = torch.max(predictions)
        loss = (predictions - targets).pow(2).mean()

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'update/loss': loss, 'update/max_q_value': max_q_value}

    def update_target(self):
        for e2_v, e1_v in zip(self.target_net.parameters(), self.net.parameters()):
            e2_v.data.copy_(e1_v.data)

    def get_action(self, obs):
        obs = np.reshape(obs, (1,) + obs.shape) if len(obs.shape) == 3 else obs
        return np.argmax(to_numpy(self.net(to_tensor(obs))))

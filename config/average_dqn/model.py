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


class ADQN:
    def __init__(self, n_ac, lr=1e-4, discount=0.99, k=10):
        self.net = Q_Net(n_ac)
        self.target_nets = []
        self.k = min(1, k)
        for _ in range(self.k):
            self.target_nets.append(Q_Net(n_ac))
        self.discount = discount
        self.n_ac = n_ac
        self.active_q = 1

        if USE_CUDA:
            self.net.cuda()
            for i in range(k):
                self.target_nets[k].cuda()

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def update(self, states, actions, rewards, next_states, dones, double=True):
        batch_size = states.shape[0]

        q_values_next_targets = []
        for i in range(self.active_q):
            q_values_next_targets.append(to_numpy(self.target_nets[i](to_tensor(next_states))))
        q_values_next_target = np.stack(q_values_next_targets, 0).mean(0)

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
        self.active_q = min(self.k, self.active_q + 1)
        for i in range(self.active_q-1, 0, -1):
            self.target_nets[i].load_state_dict(self.target_nets[i-1].state_dict())
        self.target_nets[0].load_state_dict(self.net.state_dict())

    def get_action(self, obs):
        obs = np.reshape(obs, (1,) + obs.shape) if len(obs.shape) == 3 else obs
        return np.argmax(to_numpy(self.net(to_tensor(obs))))

    def save(self, model_path, global_step=-1):
        save_dict = {"net": self.net.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "global_step": global_step}
        torch.save(save_dict, model_path)

    def load(self, model_path):
        try:
            save_dict = torch.load(model_path)
            self.net.load_state_dict(save_dict["net"])
            self.optimizer.load_state_dict(save_dict["optimizer"])
            return save_dict["global_step"]
        except Exception as e:
            print(Exception(e))
            return -1

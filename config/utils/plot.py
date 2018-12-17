import argparse
import os
import sys
import json
import uuid
import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from collections import OrderedDict


train_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../train_log'))
save_dir = os.path.join(train_path, 'fig')
os.makedirs(save_dir, exist_ok=True)


class ColorAssigner(object):
    def __init__(self):
        self.available_colors = ['mauve', 'sky blue', 'grey', 'cyan', 'pink', 'teal', 'yellow',
                                 'orange', 'magenta', 'purple', 'green', 'red', 'blue']
        self.num_colors = len(self.available_colors)
        self.items = {}

    def __call__(self, algo):
        if algo in self.items:
            return self.items[algo]
        else:
            if len(self.available_colors) == 0:
                raise ValueError('Only support {} colors'.format(self.num_colors))

            choice = self.available_colors.pop()
            self.items[algo] = choice
            return 'xkcd:' + choice


def whether_choose(algo, algo_lst):
    if len(algo_lst) == 0:
        return False
    for algo_name in algo_lst:
        if algo_name in algo:
            return True
    return False


def get_reward_from_event(args):
    reward_dict = dict()
    for filepath in glob.glob(os.path.join(train_path, args.algo+os.path.sep+args.env+":*")):
        algo = args.algo

        # add event file into a list
        events = []
        for filename in os.listdir(filepath):
            if filename.startswith("events.out."):
                events.append(filename)

        # handle each event in the event list
        for event in events:
            rewards = {}

            print('[Info]      Handle {}:{}.'.format(algo, event))
            event_rewards = {}
            try:
                for e in tf.train.summary_iterator(os.path.join(filepath, event)):
                    for v in e.summary.value:
                        if v.tag == 'reward' and e.step not in event_rewards:
                            event_rewards[e.step] = v.simple_value
            except Exception as e:
                print('[Info]      {}.'.format(Exception(e)))
            if len(event_rewards) == 0:
                continue
            rewards.update({event: event_rewards})
            reward_dict[algo] = rewards

    if args.save_data:
        out = os.path.join(save_dir, args.algo+os.path.sep+'.data.json')
        json.dump(reward_dict, open(out, 'w'))

    if args.show_info:
        print('+' + '-' * 69 + '+')
        print('+' + '-' * 69 + '+')
        print('|{:10s}  {: ^50s} {: >6s}|'.format('Algo', 'Event', 'Len'))
        for key, value in reward_dict.items():
            for k, v in value.items():
                if len(k) > 50:
                    k = k[:50]
                print('|{:10s}  {:50s} {:6d}|'.format(key, k, len(v)))
        print('+' + '-' * 69 + '+')

    return reward_dict


def plot_reward(reward_dict, title, length, algos, dpi=300,
                fig_basename=None, save=True, viz=False):
    try:
        plt.figure(figsize=(6, 6))
    except Exception as e:
        print(Exception(e))

    reward_dict = OrderedDict(sorted(reward_dict.items()))
    MEAN_LENGTH = length // 10
    ca = ColorAssigner()

    for algo, rewards in reward_dict.items():
        whether_write = whether_choose(algo, algos)
        if not whether_write:
            continue
        print(algo)

        rwds = {}
        for event_reward in rewards.values():
            rwds.update(event_reward)
        rwds = [rwds[step] for step in sorted(rwds)]
        mu = []
        upper = []
        lower = []

        for i in range(min(len(rwds), length)):
            if i < MEAN_LENGTH:
                mean = np.mean(rwds[0:i + 1])
                std = np.std(rwds[0:i + 1])
            else:
                mean = np.mean(rwds[i - MEAN_LENGTH: i])
                std = np.std(rwds[i - MEAN_LENGTH: i])
            mu.append(mean)
            upper.append(mean + 1. * std)
            lower.append(mean - 1. * std)
        mu, lower, upper = mu[:length], lower[:length], upper[:length]
        plt.plot(np.arange(len(mu)), np.array(mu), linewidth=1.0, label=algo, color=ca(algo))
        plt.fill_between(np.arange(len(mu)), upper, lower, alpha=0.3, color=ca(algo))

    plt.grid(True)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Avg_reward")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    if save:
        if fig_basename is None:
            fig_basename = title + uuid.uuid4().hex + '.png'
        plt.savefig(os.path.join(save_dir, fig_basename), dpi=dpi)

    if viz:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '--project-name', type=str, default='dqn')
    parser.add_argument('--env', '--e', '--env-name', type=str, default='Pong')
    parser.add_argument('--save_data', action='store_true', default=False)
    parser.add_argument('--show_info', action='store_false', default=True)

    parser.add_argument('--x_len', type=int, default=3000)
    FLAGS = parser.parse_args()

    r_dict = get_reward_from_event(FLAGS)
    plot_reward(r_dict, FLAGS.env, FLAGS.x_len, algos=FLAGS.algo)

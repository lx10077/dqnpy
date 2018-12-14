from train import train
from wrapper import Atari_Wrapper
from model import DQN
from util import train_dir


def main(game_name):
    assert 'NoFrameskip-v4' in game_name

    env = Atari_Wrapper(game_name)
    estimator = DQN(env.action_n, 1e-4, 0.99)
    base_path = train_dir(game_name[:-14], "DQN")

    train(env,
          estimator,
          base_path,
          batch_size=32,
          epsilon=0.01,
          update_every=4,
          update_target_every=25,
          learning_starts=50,
          memory_size=1000,
          num_iterations=20000)


if __name__ == "__main__":
    main("BreakoutNoFrameskip-v4")

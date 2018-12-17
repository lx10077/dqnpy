import click
from train import train
from wrapper import Atari_Wrapper
from model import DQN
from util import train_dir


@click.command()
@click.option('--lr', type=float, default=1e-4)
@click.option('--update_target_every', type=int, default=4)
def main(lr, update_target_every, game_name="PongNoFrameskip-v4"):
    assert 'NoFrameskip-v4' in game_name

    env = Atari_Wrapper(game_name)
    estimator = DQN(env.action_n, lr, 0.99)
    base_name = "{}:lr={}:ute={}".format(game_name[:-14], lr, update_target_every)
    base_path = train_dir(base_name)

    train(env,
          estimator,
          base_path,
          batch_size=32,
          epsilon=0.01,
          update_every=4,
          update_target_every=update_target_every,
          learning_starts=1000,
          memory_size=1000,
          num_iterations=20000)


if __name__ == "__main__":
    main()

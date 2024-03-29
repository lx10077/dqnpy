import click
from train import train
from wrapper import Atari_Wrapper
from model import ADQN
from util import train_dir


@click.command()
@click.option('--game_name', prompt='game name: (with NoFrameskip-v4)')
@click.option('--lr', type=float, default=1e-4)
@click.option('--update_target_every', type=int, default=2500)
def main(game_name, lr, update_target_every):
    if 'NoFrameskip-v4' not in game_name:
        game_name = game_name + 'NoFrameskip-v4'
    assert 'NoFrameskip-v4' in game_name

    env = Atari_Wrapper(game_name)
    estimator = ADQN(env.action_n, lr, 0.99)
    base_name = "{}:lr={}:ute={}".format(game_name[:-14], lr, update_target_every)
    base_path = train_dir(base_name)

    train(env,
          estimator,
          base_path,
          batch_size=32,
          epsilon=0.01,
          update_every=4,
          update_target_every=update_target_every,
          learning_starts=5000,
          memory_size=100000,
          num_iterations=200000000)


if __name__ == "__main__":
    main()

import argparse
import yaml

from core.trainer import Trainer


def main(args):
    config_path = args.c
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        default='configs/ave.yaml',
        type=str,
        metavar='CONFIG_PATH',
        help='Path to yaml configuration file'
    )
    args = parser.parse_args()
    main(args)

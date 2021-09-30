import argparse
import yaml

from core.tester import Tester


def main(args):
    config_path = args.c
    ckpt_path = args.p
    name = args.n
    output_dir = args.o
    device = args.d
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    tester = Tester(config, ckpt_path, output_dir, name, device)
    tester.test_checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        type=str,
        metavar='CONFIG_PATH',
        help='Path to yaml configuration file'
    )
    parser.add_argument(
        '-p',
        type=str,
        metavar='CKPT_PATH',
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '-o',
        type=str,
        metavar='OUTPUT_DIR',
        help='Path to the output images'
    )
    parser.add_argument(
        '-n',
        type=str,
        metavar='NAME',
        help='Name of the test'
    )
    parser.add_argument(
        '-d',
        type=str,
        metavar='DEVICE',
        help='GPU device'
    )
    args = parser.parse_args()
    main(args)

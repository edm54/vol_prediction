from argparse import ArgumentParser
from data import Data
import pickle as pkl
from model import Model

arg_parser = ArgumentParser(description='Control the project')

arg_parser.add_argument(
    '-l',
    help='Load Data',
    dest='load',
    action='store_true',
    default=False,
    required=False
)

arg_parser.add_argument(
    '-c',
    help='Input initial vol to network',
    dest='combined',
    action='store_true',
    default=False,
    required=False
)

arg_parser.add_argument(
    '-t',
    help='Train network, if false, load',
    dest='train',
    action='store_true',
    default=False,
    required=False
)

args = arg_parser.parse_args()

if args.load:
    spy_data = Data(None, None, skip_init=True)
    spy_data.load_data()


else:
    spy_data = Data('SPY', 'VXX')
    spy_data.save_data()

model = Model(spy_data, combined=args.combined, train=args.train)

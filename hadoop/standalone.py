import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')


    args = parser.parse_args()
    print(args.accumulate(args.integers))

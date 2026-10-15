import argparse
import json
from .workflow import train, replay

def main():
    parser = argparse.ArgumentParser(description="Stochastic warehouse navigation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the quick simulated warehouse experiment")
    training = commands.add_parser("train")
    training.add_argument("--mode", choices=["quick", "full"], default="full")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", default="outputs/full")
    training.add_argument("--map", dest="map_path", help="Optional environment JSON")
    demonstration = commands.add_parser("replay")
    demonstration.add_argument("--model", default="outputs/full/models/policies.npz")
    demonstration.add_argument("--algorithm", choices=["q_learning", "sarsa", "value_iteration"],
                               default="q_learning")
    demonstration.add_argument("--start", nargs=2, type=int, default=[0, 0], metavar=("ROW", "COLUMN"))
    demonstration.add_argument("--seed", type=int, default=999)
    demonstration.add_argument("--image", help="Optional PNG path for the sample trajectory")
    args = parser.parse_args()
    if args.command == "demo":
        train("quick", 42, "outputs/demo")
    elif args.command == "train":
        train(args.mode, args.seed, args.output, args.map_path)
    else:
        print(json.dumps(replay(args.model, args.algorithm, args.start, args.seed, args.image), indent=2))

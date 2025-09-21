"""Provide a single file for project selection and initialization."""

from project_parser import create_parser

from HousePrices.HousePricesEnvironment import HousePricesRegressionEnv
from PolymerPrediction.main import PolymerPredictionEnv
from experiment_logger import ExperimentLogger


def run_house_price_regression(args, logger):
    """Run HousePriceRegression project."""
    print(
        f"Running HousePricesRegression with:"
        f"\n  SEED={args.seed}"
        f"\n  algorithm={args.algorithm}"
        f"\n  visualize={args.visualize}"
        f"\n  submit={args.submit}"
    )

    HousePricesRegressionEnv(
        seed=args.seed, visualize=args.visualize
    ).run_experiment(algorithm=args.algorithm, submit=args.submit, seed=args.seed)

def run_polymer_prediction(args, logger):
    """Run PolymerPrediction project."""
    PolymerPredictionEnv(
        seed=args.seed
    ).run_experiment()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logger = ExperimentLogger()

    print(f"Running project: {args.project}")
    print(f"Using seed: {args.seed}")

    match args.project:
        case "house_prices":
            run_house_price_regression(args, logger)

        case "polymer_prediction":
            run_polymer_prediction(args, logger)

        case _:
            print("Invalid project name.")
            exit()

    exit()

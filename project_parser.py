import argparse


def create_parser() -> argparse.ArgumentParser:
    """Return argument parser to handle user defined parameters."""
    parser = argparse.ArgumentParser(
        description="Launch multiple ML subprojects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Project selection
    parser.add_argument(
        "project",
        choices=["house_prices", "polymer_prediction", "all"],
        help="Which project to run.",
    )

    # Common arguments
    parser.add_argument(
        "--seed", type=int, default=476, help="Random seed (default: 476)"
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: str(x).lower() in ("true", "1", "yes", "on"),
        help="Whether to enable visualization."
    )

    # House Prices arguments
    house_group = parser.add_argument_group("House Prices arguments")
    house_group.add_argument(
        "--algorithm",
        choices=["neural_net", "yggdf", "sklearn_rf", "ridge", "xgb", "skl_grb", "ensemble"],
        default="ensemble",
        help="Regression algorithm to use (default: ensemble)."
    )
    house_group.add_argument(
        "--submit", action="store_true", help="Make Kaggle submission results."
    )

    return parser

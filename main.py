from HousePrices.HousePrices import HousePricesRegressionEnv
from PolymerPrediction.main import PolymerPredictionEnv

if __name__ == '__main__':
    # TODO: ArgParser

    SEED = 476

    algorithm = [
        "NN", "yggdf", "sklearn_rf", "ridge", "grb", "ensemble"
    ]

    HousePricesRegressionEnv(
        SEED=SEED, visualize=False).run_regression(
            algorithm="ensemble", submit=True, SEED=SEED
    )

    PolymerPredictionEnv(seed).run_experiment()

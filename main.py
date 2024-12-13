from HousePrices.HousePrices import HousePricesRegressionEnv

if __name__ == '__main__':
    # TODO: ArgParser

    SEED = 476

    # HousePricesRegressionEnv(SEED=SEED).run_regression(algorithm="NN", submit=False, SEED=SEED)
    # HousePricesRegressionEnv(SEED=SEED).run_regression(algorithm="yggdf", tune=False, submit=False, SEED=SEED)
    # HousePricesRegressionEnv(SEED=SEED).run_regression(algorithm="sklearn_rf", tune=False, submit=False, SEED=SEED)
    # HousePricesRegressionEnv(SEED=SEED).run_regression(algorithm="grb", submit=False, SEED=SEED)

    HousePricesRegressionEnv(SEED=SEED).run_regression(algorithm="ensemble", submit=False, SEED=SEED)

from HousePrices.HousePrices import HousePricesRegression

if __name__ == '__main__':

    # TODO: ArgParser

    # HousePricesRegression(algorithm="NN", tune=False).run()
    # HousePricesRegression(algorithm="yggdf", tune=True, submit=True).run()
    HousePricesRegression(algorithm="ensemble", tune=True).run()

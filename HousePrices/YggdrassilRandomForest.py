import ydf
from copy import deepcopy


class YggdrassilRandomForest(ydf.RandomForestLearner):
    """
    Random Forest model with Yggdrassil implementation
    for overloading "fit" for compatibility with Stacked regression.
    """

    def __init__(self, **kwargs):
        for kwarg in kwargs:
            print(kwarg)
        kwargs.pop("label")
        super().__init__(label='SalePrice', **kwargs)

    def fit(self, dataset):
        return super().train(dataset)

    def get_params(self, deep=False):
        params = deepcopy(self.hyperparameters)
        # params.remove("deep")
        return params

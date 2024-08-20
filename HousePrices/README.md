

## House Price Prediction


1. Data exploration
    - Exploring the data to understand the features and their relationships with the target variable.
    - Visualizing the data using plots and graphs.
    - Identifying the features that are important for the prediction 
   (strongly correlated with the SalePrice) with confusion matrices.
    - Relationships between the features themselves.
   

2. Preprocessing data by:
    - Removing outliers
    - Handling missing values
      - Imputing column with missing values
      - or filling them with specified values
    - Encoding categorical variables
      - One-hot encoding
      - Label encoding


3. Model selection
    - TensorFlow Decision Trees: Random Forest
    - Yggdrasil: Random Forest
    - Scikit-learn: Random Forest
    - Tensorflow: Neural Network

   - And hyperparameter tuning the above models with Random search

   - Ensemble learning with
     - Ridge Regression
     - Gradient Boosting
     - xgboost
     - LightGBM
     - Random Forest
     - and a StackingCVRegressor 


4. Model evaluation

    - Mean Squared Error
    - R2


5. Results
   
   - Both the Random Forest and Neural Network models with the best parameters 
achieve only about 80–90% R2 accuracy on the test data, and a Kaggle public score 
of about 0.147+.
   - The ensemble model outperforms every non-stacked model with a Kaggle public score of 0.12+.



TODO:

Skewness and Kurtosis fix.
Log (or Johnson) transformation of the target variable.
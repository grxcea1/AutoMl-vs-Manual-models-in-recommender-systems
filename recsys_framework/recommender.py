import numpy as np #importing numpy for mathematical operations
import pandas as pd # importing pandas for building dataframes
import time #importing time to measure how long training takes

#evaluation metrics to evaluate model performance
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              precision_score,
                              recall_score,
                              f1_score)


#recommender class with different modes (accurate, fast, explainable)
class Recommender:
    
    VALID_MODES = ('accurate', 'fast', 'explainable')  #the allowed model types

    #constructor (initialises the model)
    def __init__(self, mode='accurate',
                 time_budget=120):
        
        #checks if the selected mode is valid
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}") #prints this error message if the mode is not valid
        
        self.mode = mode  #selected mode
        self.time_budget = time_budget  #time for AutoML search
        self.model = None  #placeholder for trained model
        self.feature_names_in_ = None  #stores training feature names
        self.trained = False  #checks if model has been trained
        self.runtime = None  #stores training time

    #train function (trains model depending on mode)
    def train(self, X_train, y_train):
        
        self.feature_names_in_ = X_train.columns.tolist()  # save feature names for later
        start = time.time()  # start timer

        # ACCURATE uses AutoML to find best model (FLAML had the highest accuracy in my tests)
        if self.mode == 'accurate':
            from flaml import AutoML
            
            self.model = AutoML()  #creates AutoML model
            
            # trains and searches for best model within time limit
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                task='regression',  # predicting ratings
                time_budget=self.time_budget,
                metric='rmse',  # optimise for accuracy
                verbose=0  # no training logs
            )
            
            print(f"  FLAML selected: "
                  f"{self.model.best_estimator}")  #prints best model

        #FAST uses neural network (as mlp was the fastest in my tests while still having good accuracy)
        elif self.mode == 'fast':
            from sklearn.neural_network import (
                MLPRegressor)
            
            self.model = MLPRegressor(
                activation='relu',  #adds non-linearity (positive satys the same , negatives becomes 0)
                max_iter=300,  #max training iterations
                random_state=42,  #ensures same results each run
                early_stopping=True,  #stops if no improvement
                validation_fraction=0.1,  #10% used for validation
                verbose=False  # runs quietly, reduces clutter in output
            )
            
            self.model.fit(X_train, y_train)  #trains model

        #EXPLAINABLE uses decision tree as it is simple and easy to interpret
        elif self.mode == 'explainable':
            from sklearn.tree import (
                DecisionTreeRegressor)
            
            self.model = DecisionTreeRegressor(
                random_state=42)  #simple interpretable model
            
            self.model.fit(X_train, y_train)  #trains model

        self.runtime = time.time() - start  #calculates training time
        self.trained = True  #marks as trained
        
        print(f"  [{self.mode}] trained in "
              f"{self.runtime:.1f}s")  #prints runtime
        
        return self

    # predict function is what generates predictions
    def predict(self, X):
        self._check_trained()  #make sure model is trained first
        X = self._align(X)  #aligns features with training data
        return self.model.predict(X) #returns predictions

    #recommend function (returns top 10 recommendations)
    def recommend(self, user_id, preprocess, k=10):
        self._check_trained()

        #gets user data
        user_row = preprocess.users[
            preprocess.users['user_id'] == user_id]
        
        #if user not found it will throw an error
        if user_row.empty:
            raise ValueError(
                f"user_id {user_id} not found.")

        #gets items user has already rated
        rated = preprocess.ratings[
            preprocess.ratings['user_id'] == user_id
        ]['item_id'].unique()

        #gets all items
        all_items = preprocess.items['item_id'].unique()

        #gets items not yet rated
        candidates = [i for i in all_items
                      if i not in rated]
        
        if not candidates:
            return []  # #if there are no unrated items, return an empty recommendation list


        #creating a dataframe for predictions
        df = pd.DataFrame({'item_id': candidates})
        df['user_id'] = user_id

        #merging user and item features to create input data for rating prediction
        df = df.merge(preprocess.users,
                      on='user_id', how='left')
        df = df.merge(preprocess.items,
                      on='item_id', how='left')

        titles = df['title'].values  #saving titles before dropping

        #columns not needed for model
        drop_cols = ['timestamp', 'zip', 'title',
                     'release_date',
                     'video_release_date', 'IMDb_URL']
        
        # only drop columns that exist
        drop_cols = [c for c in drop_cols
                     if c in df.columns]
        
        df = df.drop(columns=drop_cols).dropna()  # remove unnecessary columns and missing values

        df['gender'] = df['gender'].map(
            {'M': 0, 'F': 1})  # convert gender to numeric

        df = pd.get_dummies(df, columns=['occupation'])  #one-hot encode occupation

        df = self._align(df)  #align with training features

        if df.empty:
            return []  #no valid data left after preprocessing

        preds = self.model.predict(df)  #makes predictions

        top_idx = np.argsort(preds)[::-1][:k]  #gets top 10 predictions

        return [(titles[i], float(preds[i]))
                for i in top_idx if i < len(titles)]  #returns title + score

    # evaluation function (calculates performance metrics)
    def evaluate(self, X_test, y_test):
        self._check_trained()

        y_pred = self.predict(X_test)  #makes predictions

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)

        # convert to binary (>=4 = liked)
        y_bin_t = (np.array(y_test) >= 4).astype(int)
        y_bin_p = (np.array(y_pred) >= 4).astype(int)

        #returns metrics
        return {
            'MSE': float(mse),
            'RMSE': rmse,
            'MAE': float(mae),
            'Precision': float(precision_score(
                y_bin_t, y_bin_p, zero_division=0)),
            'Recall': float(recall_score(
                y_bin_t, y_bin_p, zero_division=0)),
            'F1': float(f1_score(
                y_bin_t, y_bin_p, zero_division=0)),
            'Runtime': self.runtime
        }

    # align function to match test data with training features
    def _align(self, X):
        if self.feature_names_in_ is None:
            return X  #if no training features were stored, return the data the same way
        
        X = X.copy()  #makes a copy of the data so the original doesnt get changed

        # add missing columns with 0
        for col in self.feature_names_in_:
            if col not in X.columns:
                X[col] = 0

        return X[self.feature_names_in_]   #makes sure the columns are in the correct order

    #this function makes predictions are not made before the model is trained
    def _check_trained(self):
        if not self.trained:
            raise RuntimeError(
                "Call train() first.")  # error if not trained
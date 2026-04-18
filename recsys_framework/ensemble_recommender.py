import numpy as np  #importing numpy for mathematical operations
import pandas as pd #import pandas for builidng dataframes
import time #importing time to work out the time each model takes to train
from flaml import AutoML #Automl library
from sklearn.neural_network import MLPRegressor #neural network
#evaluation metrics to evaluate the ensemble model
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              precision_score,
                              recall_score,
                              f1_score)


#this is the class for the ensemble model (combines MLP with FLAML)
class EnsembleRecommender: 

    # this is the contructor (it initialises the model)
    def __init__(self, flaml_weight=0.65, #FLAML has a higher weight then MLP (due to greater accuracy) it should have a greater influence on the ensembles final predictions
                 mlp_weight=0.35, #lower weight
                 time_budget=120):#time given for flaml to find best model
        
        self.flaml_weight = flaml_weight #higher weight for FLAML predictions 
        self.mlp_weight = mlp_weight #weight for MLP predictions
        self.time_budget = time_budget #time limit for the entire search process including training
        self.flaml_model = None   #placeholder for trained FLAML model
        self.mlp_model = None #placeholder for trained MLP model
        self.feature_names_in_ = None #stores the feature names from training (which is important for consistency)
        self.trained = False  #checks if model has been trained
        self.runtime = None  #stores the training runtime

    # this is the train function that will train both models
    def train(self, X_train, y_train):
        self.feature_names_in_ = X_train.columns.tolist() #saves the feature names (its needed later to align the test data)
        start = time.time() #start timer

        print("Ensemble Training FLAML...")
        self.flaml_model = AutoML() #creates Automl model

        #trains Automl model and finds the best model automatically
        self.flaml_model.fit( 
            X_train=X_train,
            y_train=y_train,
            task='regression',   #preditics the ratings 
            time_budget=self.time_budget,
            metric='rmse', #optimise based on accuracy
            verbose=0
        )
        print(f"Ensemble FLAML selected: "  
              f"{self.flaml_model.best_estimator}") #shows best model chosen by FLAML

        print("  [Ensemble] Training MLP component...")

        #trains MLP model 
        self.mlp_model = MLPRegressor(
            activation='relu',    #applies ReLU function to introduce non-linearity (positive values stay positive and negative values become 0)
            max_iter=300,         #max training iterations
            random_state=42,      #makes sure results are reporducible
            early_stopping=True,  #stops if there is no improvement
            validation_fraction=0.1,  #10% used for validation
            verbose=False              #controls how much is printied
        )
        self.mlp_model.fit(X_train, y_train)   # trains neural network
        self.runtime = time.time() - start  #calculates total training time
        self.trained = True #marks model as trained
        print(f"Ensemble Trained in "
              f"{self.runtime:.1f}s") #prints runtime
        return self

    #the predict function combines the predictions made by both the models
    def predict(self, X):
        self._check_trained()  #checks if model is trained first
        X = self._align(X)#aligns features with training data
        flaml_preds = self.flaml_model.predict(X) #gets predictions from FLAML model
        mlp_preds = self.mlp_model.predict(X)  #gets predictions from MLP model

        #combines predictions using the weighted averages for the ensemble
        return (self.flaml_weight * flaml_preds
                + self.mlp_weight * mlp_preds)
    


    #recommend function that gives the top 10 recommendations based on the semble predictions
    def recommend(self, user_id, preprocess, k=10):
        self._check_trained() #checks if model is trained first

        user_row = preprocess.users[
            preprocess.users['user_id'] == user_id] #gets the user data
        if user_row.empty: #if the user is not found it throws an error
            raise ValueError(
                f"user_id {user_id} not found.")

        rated = preprocess.ratings[
            preprocess.ratings['user_id'] == user_id 
        ]['item_id'].unique()#gets the items that are already rated by the user

        all_items = preprocess.items['item_id'].unique() #gets all the items
        candidates = [i for i in all_items
                      if i not in rated] #selects items user hasn't rated 
        if not candidates:
            return [] #if there are no unrated items, return an empty recommendation list

        #creating a dataframe for predictions
        df = pd.DataFrame({'item_id': candidates})
        df['user_id'] = user_id

        #merging user and item features to create input data for rating prediction
        df = df.merge(preprocess.users,
                      on='user_id', how='left')
        df = df.merge(preprocess.items,
                      on='item_id', how='left')
        titles = df['title'].values #saving titles before dropping

        #columns not needed for the model
        drop_cols = ['timestamp', 'zip', 'title',
                     'release_date',
                     'video_release_date', 'IMDb_URL']
        
        #only dropping existing columns
        drop_cols = [c for c in drop_cols
                     if c in df.columns]
        #dropping unnecessary columns and missing values
        df = df.drop(columns=drop_cols).dropna()

        #Converting gender to numeric
        df['gender'] = df['gender'].map(
            {'M': 0, 'F': 1})
        
        #one-hot encode occupation
        df = pd.get_dummies(df, columns=['occupation'])

        #align with training features
        df = self._align(df)

        #returns an empty list if no valid data remains after preprocessing
        if df.empty:
            return []

        #makes predictions using ensemble model
        preds = self.predict(df)

        #gets indices of top 10 predictions
        top_idx = np.argsort(preds)[::-1][:k]

        #returns top recommendations (title + predicted score)
        return [(titles[i], float(preds[i]))
                for i in top_idx if i < len(titles)]

    #evaluation function that calculates the metrics for ensemble model
    def evaluate(self, X_test, y_test):
        self._check_trained() #checks if model is trained first
        y_pred = self.predict(X_test) #predicts ratings

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)

        #converts ratings to binary (>=4 = liked)
        y_bin_t = (np.array(y_test) >= 4).astype(int)
        y_bin_p = (np.array(y_pred) >= 4).astype(int)

        #returns evaluation metrics
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
    

    #align function to make sure the test features are the same as the training features    
    def _align(self, X):
        
        #if no training features were stored, return the data the same way
        if self.feature_names_in_ is None:
            return X
        
        #makes a copy of the data so the original doesnt get changed
        X = X.copy()

        #adds missing columns with the value 0
        for col in self.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        #makes sure the columns are in the correct order
        return X[self.feature_names_in_]

    #this function makes predictions are not made before the model is trained
    def _check_trained(self):
        if not self.trained:
            raise RuntimeError(
                "Call train() first.") #it will throw this error

    
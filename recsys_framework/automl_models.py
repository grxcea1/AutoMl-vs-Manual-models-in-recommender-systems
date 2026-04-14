from flaml import AutoML  #this imports the flaml automl tool that automatically finds and trains the best model
import pandas as pd #pandas handles data 
import h2o #loads the main h20 library
from h2o.automl import H2OAutoML #imports the Automl tool inside h20

#This class is the container for all the automl models 
class AutoMLModels:


    #this function will let flaml automatically find and train the best model using the data given within a certain time frame
    def flaml(self, X_train, y_train, time_budget=120):
        model = AutoML() #this creates an empty AutoML object to be used later

        #this trains the model
        model.fit(
            X_train = X_train, #these are the features the model learns from
            y_train = y_train, #these are the ratings to predict 

            task = "regression", # this tells the model what type of task it is (regression - predicts a number (rating between 1-5)) , classification would predict a category
            time_budget = time_budget, #time limit for the model to search for the best model to use 

            metric = "rmse", #flaml optimises for lowest rmse (better predictions)
            verbose = 0 #does not print logs so terminal stays clean
        )

        #this line prints out the model that flaml chose
        print(f" FLAML selected: {model.best_estimator}") #this prints the best model flaml found for the task 
        return model #returns trained model
    





    #pycaret Automl method 
    def pycaret(self, X_train, y_train):
        from pycaret.regression import(setup, #this intialises the pycaret enviroment (this makes sure the pycaret is ready to run the models)
                                       compare_models, #trains and compares many models
                                         pull) # gets the comparison results as a table 
        
        # this combines the training features and the target into one dataframe (format needed for pycaret)
        train_df = X_train.copy() 
        train_df['rating'] = y_train.values

        #this sets up the pycaret enviroment
        setup(
            data=train_df,
            target='rating', #this is the column we want to predict
            session_id=42, # this controlds the randomness (so we get the same results everytime)
            verbose=False, #this makes the program print little (makes output clean and readable)
            html=False #disables html output
        )

        #trains and compares multiple models automatically 
        best_model = compare_models(
            sort='RMSE',  #ranks the models by rmse (lower rmse = high accuracy)
            n_select=1, #selects the besy model
            verbose=True #shows progress intead of hiding it
        )

        results_df = pull() # gets comparison table 
        best_name = str(results_df.index[0]) #gets the name of the best model
        print(f"  PyCaret selected: {best_name}") #prints the name of the best model
        return best_model #returns the best trained model
    



    #h20 Automl method
    def h2o(self, X_train, y_train, max_models=10):
        
        # starts h20 server 
        h2o.init(verbose=False)
        h2o.no_progress()

        train_df = self.undummy_occupation(X_train.copy()) #converts the dummy occupation columns back to a single column
        train_df['rating'] = y_train.values #adds a target column
        train_h2o = h2o.H2OFrame(train_df) #converts pandas dataframe into h20 format

        #this defines the features to train and the targets
        feature_cols = X_train.columns.tolist()
        target_col = 'rating'

        #this creates a h20 Automl object
        aml = H2OAutoML(
            max_models=max_models, #max number of models to try
            seed=42, #controls randomness
            sort_metric='RMSE' #optimises for rmse
        )

        # Trains model
        aml.train(
            x=feature_cols, #input fetures 
            y=target_col, #target variable
            training_frame=train_h2o
        )


        #this gets and prints the best model
        best_name = aml.leader.algo
        print(f"  H2O selected: {best_name}")

        class H2OWrapper: #this creates an object that wraps h20

            def __init__(self, leader):
                    self.leader = leader #trained h20 model
                    self.best_name = best_name #name of the best model
                    self.feature_names_in_ = X_train.columns.tolist() #column names used in training

            @staticmethod #helper function hats does not use the object itself
            def _undummy(df): #this functions converts the occupation columns from one-hot encoded columns done in preropcessing back to one column (so it can be used in h20)
                occ_cols = [c for c in df.columns if c.startswith("occupation_")] #finds all the columns that start with occupation 
                if occ_cols:
                    df = df.copy() #makes a copy to not mess up original data

                    # for each row it finds out what occupation column has the value one
                    df["occupation"] = (
                        df[occ_cols]
                        .idxmax(axis=1) #gets the column name with the value 1
                        .str.replace("occupation_", "", regex=False)#cleans the name by removing "occupation" and replacing it with nothing ("")
                    )
                    df = df.drop(columns=occ_cols)  # removes the original dummy columns so we only have one occupation column
                return df #returns the updated dataframe
 
            def predict(self, X): #functions to make predictions
                frame = h2o.H2OFrame(X) #converts pandas to h20 format
                preds = self.leader.predict(frame)  #gets predictions
                return preds.as_data_frame(
                        )['predict'].values #converts predictions back to numpy array
            
        return H2OWrapper(aml.leader) #returns wrapped model so its esy to use later 















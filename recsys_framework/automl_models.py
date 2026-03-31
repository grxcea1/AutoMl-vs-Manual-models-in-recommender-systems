#this class acts as a container holding all my automl models
class AutoMLModels:


    #this function lets flaml automatically find and train the best model using the data given within a certain time frame
    def flaml(self, X_train, y_train, time_budget=120):
        from flaml import AutoML  #this imports the flaml automl tool that automatically finds and trains the best model
        model = AutoML() # this creates an empty AutoML object

        model.fit(
            X_train = X_train, #these are the features the model learns from
            y_train = y_train, #these are the ratings to predict 

            task = "regression", # this tells the model what type of task it is (regression - predicts a number (rating between 1-5)) , classification would predict a category
            time_budget = time_budget, #time limit given to the model to search for the best model to use 

            metric = "rmse", #flaml optimises for lowest rmse (ensures fair comparison as i will use it for all models)
            verbose = 0 #prints nothing in output at first so terminal stays clean


        )

        print(f" FLAML selected: {model.best_estimator}") #this prints the best model flaml found for the task 
        return model
    


    def pycaret(self, X_train, y_train):
        from pycaret.regression import(setup, #this intialises the pycaret enviroment (this makes sure the pycaret is ready to run the models)
                                       compare_models, #trains and ranks many algorithms
                                         pull) #gets the comparison results as a table 
        
        import pandas as pd #this is needed to create the combined dataframe 
        train_df =X_train.copy() #this creates a copy of the x_train data
        train_df['rating'] = y_train.values


        setup(
            data=train_df,
            target='rating',
            session_id=42,
            verbose=False,
            html=False
        )



        best_model = compare_models(
            sort='RMSE',
            n_select=1,
            verbose=True
        )



        results_df = pull()
        best_name = str(results_df.index[0])
        print(f"  PyCaret selected: {best_name}")


        return best_model
    


    def h2o(self, X_train, y_train, max_models=10):

        import h2o
        from h2o.automl import H2OAutoML
        import pandas as pd
        
        h2o.init(verbose=False)
        h2o.no_progress()

        train_df = X_train.copy()
        train_df['rating'] = y_train.values
        train_h2o = h2o.H2OFrame(train_df)


        feature_cols = X_train.columns.tolist()
        target_col = 'rating'

        aml = H2OAutoML(
            max_models=max_models,
            seed=42,
            sort_metric='RMSE'
        )


        aml.train(
            x=feature_cols,
            y=target_col,
            training_frame=train_h2o
        )


        best_name = aml.leader.algo
        print(f"  H2O selected: {best_name}")



        class H2OWrapper:

            def __init__(self, leader):
                    self.leader = leader
                    self.best_name = best_name
                    self.feature_names_in_ = X_train.columns.tolist()

            def predict(self, X):
                frame = h2o.H2OFrame(X)
                preds = self.leader.predict(frame)
                return preds.as_data_frame(
                        )['predict'].values
            
        return H2OWrapper(aml.leader)















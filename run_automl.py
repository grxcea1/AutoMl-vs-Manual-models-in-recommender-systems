from recsys_framework.preprocessing import preprocessing #importing preprocessing class into my folder
from sklearn.metrics import mean_squared_error, mean_absolute_error #importing the mse and Rmse from the sklearn library
from sklearn.metrics import f1_score, recall_score, precision_score  #importing f1,precison and recall from the sklearn library
import numpy as np #importing numpy for mathematical operations
import time #importing time to work out the time each model takes to train
import matplotlib #load the matploy library
matplotlib.use("agg") #'agg' is needed for when i want to save files
import matplotlib.pyplot as plt #plt will be used to refer matplot library (for making graphs)
import pandas as pd #import pandas for builidng dataframes
from pycaret.regression import (setup, compare_models, #imports pycaret tools for regression AutoML
                                    predict_model, pull)
import h2o
from h2o.automl import H2OAutoML







#function to evaluate each model
def evaluate_model(y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred) #this caluculates the error between predictions (it gets squared so that the negatives and positives dont cacel eachother out)
  rmse = np.sqrt(mse) #sqrt of mse (this is so that it is the same scale as the ratings)(lower = more accurate)
  mae = mean_absolute_error(y_test, y_pred) #this s the average error with no penalisation on wrong answers

  # this bit changes ratings into yes or no (binary 0/1) - precison,recall and f1 can only work on binary
  y_test_binary = (np.array(y_test) >= 4).astype(int) #so if rating is 4 or 5 (liked it)
  y_pred_binary = (np.array(y_pred) >= 4).astype(int) #if rating is 3-1 (didnt like it)

  precision = precision_score(y_test_binary, y_pred_binary, zero_division= 0) # this works out, of all the that i collected which ones were good
  recall = recall_score(y_test_binary, y_pred_binary, zero_division=0) #this measures of all the good movies, how many did the recommender collect
  f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0) #this gives a combined score of precison and recall

  return mse, rmse, mae, precision, recall, f1








#this functions role is to print my results 
def print_results(model_name, runtime, mse, rmse, mae, precision, recall, f1, prec_at_k, rec_at_k):
  print("\n" + "-" *46) # this just creats a divider for readability 
  print(f"Model: {model_name}") #prints the name of the model shwoing the results
  print("-" * 46) #creates the repeat divider 

  print(f"Runtime:                  {runtime: .2f} seconds")# print runtime to 2dp
  print(f" MSE:                     {mse:.4f}") # print MSE to 4dp
  print(f" RMSE:                    {rmse:.4f}") # print RMSE to 4dp
  print(f" MAE:                     {mae:.4f}") #print MAE to 4dp
  print(f" Precision (full test) :  {precision :.4f}") #print precison to 4dp
  print(f" Recall (full test):      {recall:.4f}") #print recall  to 4dp
  print(f" F1 Score:                {f1:.4f}") #print f1 to 4dp
  print("-" * 46) #creates the repeat divider
  print(f" Precision@10 :           {prec_at_k :.4f}") #print precison@10 to 4dp
  print(f" Recall@10 :              {rec_at_k:.4f}") #print recall@10  to 4dp







#function to show the top 10 Recommendations per user
def show_top10(model, preprocess, model_name, X_test, y_test, sample_users_ids = None):
  print(f"\n  TOP 10 RECOMMENDATIONS  ({model_name})") #title
  k = 10

  if sample_users_ids is None:  #if no specific users are given, it will use the first 3 users in the dataset 
    sample_users_ids = preprocess.ratings["user_id"].unique()[:3] #this line gets all the unqiue users and takes the first 3

  for user_id in sample_users_ids: #loops though each sample user one at a time (so it will repeat 3 times since we chose 3)
    all_items = preprocess.items["item_id"].unique() # this line gets every movie id that exists in the dataset

    #the next code is what gets us a list of movies the user has already seen/rated
    rated_items = preprocess.ratings[ #looks inside ratings table
      preprocess.ratings["user_id"] == user_id #this keeps only rows for this user
    ]["item_id"].unique()# takes just the movie ids(unqiue removes duplicates)

    unrated_items = [i for i in all_items if i not in rated_items] #loops though all the movie ids and performs the operation as long as the movie is not already rated
    user_row = preprocess.users[
        preprocess.users["user_id"] == user_id] # making sure the user exists in the user dataset
    if user_row.empty:
      continue #skip if the user doesnt exist as it will have no data

    user_data = pd.DataFrame({'item_id': unrated_items})
    user_data['user_id'] = user_id
    user_data = user_data.merge(preprocess.users, on= "user_id", how="left") # this adds user details to the data 
    user_data = user_data.merge(preprocess.items, on= "item_id", how="left") # this adds movies details the data

    titles = user_data["title"].values #this just saves the movie titles so we can display them later

    drop_cols = ['timestamp', 'zip', 'title', 'release_date', 'video_release_date'] #this code just removes irrelevant columns
    drop_cols = [c for c in drop_cols if c in user_data.columns]
    user_data = user_data.drop(columns=drop_cols)# this is necessary so we dont get an error if the column is already gone
    user_data = user_data.dropna() # remove any missing values
    user_data['gender'] = user_data['gender'].map({'M': 0, 'F': 1}) # convert gender to 1 and 0
    user_data = pd.get_dummies(user_data, columns=['occupation']) # makes occupations into binary form

    #this makes sure that training column order matches models expectations
    trained_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None #if the models reme,bers column name then it should use them if not we set the column names to None
    if trained_cols is not None:   # continues if we have column names to use
      for col in trained_cols:
                if col not in user_data.columns:
                    user_data[col] = 0  #this adds the column if the column is missing 

      user_data = user_data[trained_cols] #this reorders columns to match the training column order
    if user_data.empty:# skip the user if the dataframe is empty
            continue  
    
    predicted= model.predict(user_data) #model makes predictions on unrated movies based on what it has learned
    top10_idx = np.argsort(predicted)[::-1][:10] #use the index positions of the movies to display the movie names
    top10_titles = [titles[i] for i in top10_idx if i < len(titles)] # this checks if the index is valid ( it prevents the index out of range error )

    # prints the movie titles as a numbered list , just makes it easier to read
    top10_scores = [predicted[i] for i in top10_idx if i <len(titles)] #this prints top 10 movies 

    print("\n" + "-" * 46) #line to seperate, so its ready clearly
    print(f"  User {user_id} — Top 10 Recommendations") #prints the user id and the lable to know what the list is 
    print("-" * 46) #line to seperate, so its ready clearly

    for rank, (title,score) in enumerate( #loops through titles, scores and ranking 
        zip(top10_titles, top10_scores), start=1):
        print(f"   {rank:2}. {title: <55} (predicted rating:{score:.1f}/5)") #this prints the numbered list (with the movie titles and the predicted rating)
    print("-" * 46) #line to seperate, so its ready clearly

  all_precisions = [] #stores precision at 10 for every test user (the 3 unique ones)
  all_recalls = [] #stores recalls at 10 for every test user (the 3 unique ones)

  #this loops through the 3 users in the test set
  for user_id in X_test["user_id"].unique():
    test_indices = X_test.index[X_test["user_id"] == user_id] #this collects all the rows of the unique users
    user_test_ratings = y_test.loc[test_indices] #their actual ratings from the test data
    user_test_item_ids = X_test.loc[test_indices, "item_id"] #this is the movie ids the actual ratings belong to

    actual_ratings_connect = dict(zip(user_test_item_ids, user_test_ratings)) #this creates a dictionary that matchs the movies(item) to the real ratings
    actually_liked = set(item_id for item_id, rating in actual_ratings_connect.items() if rating >=4) #movies they actually liked get stored in a set

    if len(actually_liked) == 0:
        continue  # this skips users with no liked movies in the test set

    all_items = preprocess.items["item_id"].unique() # this gets the unique movie (item) ids
    rated_items = preprocess.ratings[preprocess.ratings["user_id"] == user_id]["item_id"].unique() #gets all the item the user has already rated

    unrated_items = list(user_test_item_ids)

    user_row = preprocess.users[preprocess.users["user_id"] == user_id] # this line gets all the user data (so all the data in the different columns)
    if user_row.empty: #if the user data is missing this line skips the user
      continue

    user_data = pd.DataFrame({'item_id': unrated_items}) #this creates a dataframe of all unseen movies(items)
    user_data['user_id'] = user_id #this adds user ids to each row so the model knows who the recommendation is for
    user_data = user_data.merge(preprocess.users, on="user_id", how="left") #adds the user features to the user_data dataframe
    user_data = user_data.merge(preprocess.items, on="item_id", how="left") #adds the item feature to the user_data dataframe
    candidate_item_ids = user_data['item_id'].values #this saves the item ids before dropping columns

    drop_cols = ['timestamp', 'zip', 'title', 'release_date', 'video_release_date'] #these are the columns i dont want in the model
    drop_cols = [c for c in drop_cols if c in user_data.columns] #this makes sure only the columns that exist are kept
    user_data = user_data.drop(columns=drop_cols) #this removes unnecessary columns
    user_data = user_data.dropna() #this removes rows with missing values
    user_data['gender'] = user_data['gender'].map({'M': 0, 'F': 1}) #convet genders to numbers so the model can use it
    user_data = pd.get_dummies(user_data, columns=['occupation']) #this turns categories into binary columns
 
    trained_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None #this gets the columns that the training set uses
    if trained_cols is not None: #this makes sure the model only runs if we have the correct feature names
      for col in trained_cols:
        if col not in user_data.columns:
          user_data[col] = 0 #fill missing columns with 0 (this makes sure the input matches the training data)
      user_data = user_data[trained_cols] #this reorders the columns to match the training columns exactly
 
    if user_data.empty:
      continue # if theres no data left it will skip the user

    predicted = model.predict(user_data)  #predicts ratings for unseen items
    top_k_idx = np.argsort(predicted)[::-1][:k] #this sorts the predictions from highest to lowest (this is out of the best recommendations)
    top_k_item_ids = [candidate_item_ids[i] for i in top_k_idx if i < len(candidate_item_ids)] #this convert indices back to actually item ids
 
    hits = len([item for item in top_k_item_ids if item in actually_liked]) #this calculates how many in top k were actually liked by the users
    all_precisions.append(hits / k) #precision@10 = hits / 10 (basically how many of top 10 were relevent)
    all_recalls.append(hits / len(actually_liked)) #recall@10 = hits / total movies user actually likes (how many of the users liked items were found)
 
  avg_prec_k = np.mean(all_precisions) if all_precisions else 0.0 #calculates average precision@10 across all test users
  avg_rec_k = np.mean(all_recalls) if all_recalls else 0.0 # calculates average recall@10 across all test users
 
  return avg_prec_k, avg_rec_k #return both so run() can store them in results dictionary







def run_flaml(X_train, X_test, y_train, y_test,
              preprocess, time_budget=120):
    from flaml import AutoML  #imports FLAML Automl framework 

    #header section
    print("\n" + "=" * 46)
    print("Running FLAML AutoML...")
    print(f"Time budget: {time_budget} seconds")
    print("=" * 46)

    automl = AutoML() #creates Flaml Automl object
    start = time.time() #this starts timer to measure total training time

    # train multiple models automatically and picks the best one
    automl.fit(
        X_train=X_train, #features used for training
        y_train=y_train, #targets
        task='regression', #predicts ratings (continuous values)
        time_budget=time_budget, #max time allowed for model search 
        metric='rmse', #optimisation metric
        verbose=0 #makes sure logs dont print
    )

    runtime = time.time() - start #calculates total training time
    best_name = automl.best_estimator #best model picked by Flaml
    print(f"\nFLAML selected: {best_name}")

    y_pred = automl.predict(X_test) #predictions on test data

    # collects the values for each metric for flaml
    mse, rmse, mae, precision, recall, f1 = evaluate_model(
        y_test, y_pred)

    best_model = automl.model.estimator #decideds on the trained best model

    # this works out the top 10 metrics
    prec_k, rec_k = show_top10(
        best_model, preprocess, 'FLAML',
        X_test, y_test, sample_users_ids=[1, 2, 3])
    
    #print full results
    print_results('FLAML', runtime, mse, rmse, mae,
                  precision, recall, f1, prec_k, rec_k)
    
    #this returns all the metrics that have been calculated 
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime,
        'Precision@k': prec_k, 'Recall@k': rec_k,
        'Best Algorithm': best_name
    }









def run_pycaret(X_train, X_test, y_train, y_test, preprocess):

    #header section
    print("\n" + "-" * 46)
    print("Running PyCaret AutoML...")
    print("-" * 46)

    # combines features and target because PyCaret requires one dataset
    train_df = X_train.copy()
    train_df['rating'] = y_train.values

    start = time.time() #this starts timer to measure total training time and selection time 

    #initialise pyCaret enviroment 
    setup(
        data=train_df, #full dataset (features and target)
        target='rating', #this is the column we want to predict
        session_id=42, # this controlds the randomness (so we get the same results everytime)
        verbose=False, #this makes the program print little (makes output clean and readable)
        html=False #no html output
    )

    #traisn multiple models and selects the best one based on rmse
    best = compare_models( 
        sort='RMSE', #sorts preictions br rmse (lowest to highest)
        n_select=1, #returns only the best model 
        verbose=False # makes sure the output is clean
    )

    runtime = time.time() - start #stops the timer 
    results_df = pull() #gets the results table
    best_name = str(results_df.index[0]) # gets the best models name
    print(f"\nPyCaret selected: {best_name}") #prints the name of the best model

    test_df = X_test.copy() #makes a copy of test data so we dont mess up the original 
    predictions_df = predict_model(best, data=test_df) #uses the best model to make predictions on the test data
    y_pred = predictions_df['prediction_label'].values #this extracts the predicted values 

    mse, rmse, mae, precision, recall, f1 = evaluate_model(
        y_test, y_pred) #this evalues the moodel (compares the predictions made to the real ratings)

    
    prec_k, rec_k = show_top10(  #this is calling my function to get top top 10 recommendations per user
        best, preprocess, 'PyCaret', # best is my train pycaret modle, preprocesss is an object , pycaret is a label
        X_test, y_test, sample_users_ids=[1, 2, 3]) #the features, the actual ratings and the users 

    print_results('PyCaret', runtime, mse, rmse, mae,
                  precision, recall, f1, prec_k, rec_k) #prints the results

    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime,
        'Precision@k': prec_k, 'Recall@k': rec_k,
        'Best Algorithm': best_name
    } #this returns a dictionary that stores all the results






def undummy_occupation(df): #this functions converts the occupation columns from one-hot encoded columns done in preropcessing back to one column (so it can be used in h20)
    occ_cols = [c for c in df.columns if c.startswith("occupation_")] #finds all the columns that start with occupation 
    if occ_cols:
        df = df.copy() #makes a copy to not mess up original data

        # for each row it finds out what occupation column has the value 1
        df["occupation"] = (
            df[occ_cols]
            .idxmax(axis=1) #gets the column name with the value 1
            .str.replace("occupation_", "", regex=False) #cleans the name by removing "occupation" and replacing it with nothing ("")
                    )
        
        df = df.drop(columns=occ_cols) # removes the original dummy columns so we only have one occupation column
    return df #returns the updated dataframe




def run_h2o(X_train, X_test, y_train, y_test,
            preprocess, max_models=10):

    print("\n" + "-" * 46)
    print("Running H2O AutoML...")
    print(f"Training up to {max_models} models")
    print("-" * 46)

    h2o.init(verbose=False) #starts h20 server
    h2o.no_progress() #hides the progress bars for a cleaner output

    train_df = undummy_occupation(X_train.copy()).reset_index(drop=True) # converts one-hopt encoded occupation solumns back to a single column
    train_df['rating'] = y_train.values #adds target column 

    test_df = undummy_occupation(X_test.copy()).reset_index(drop=True) #prepares test data
    test_df['rating'] = 0  #target column (h20 needs training and test data to  have the same structure)

    # changes the training and  test data into h20 format
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    
    for col in train_h2o.columns: #loops through every column in the trianing dataset
        if train_h2o[col].isfactor()[0] or train_df[col].dtype == object: # checks if column is already cetergorical in h20
            train_h2o[col] = train_h2o[col].asfactor() # converts traning column into categorical type (no longer meaningles but now part of a group)
            test_h2o[col] = test_h2o[col].asfactor() # converts test colun into categorical type 

    feature_cols = [c for c in train_df.columns if c != 'rating'] #defines the feature columns
    target_col = 'rating' #defines the target column

    #this initialises the h20 Automl object with speciific parameters 
    aml = H2OAutoML(max_models=max_models, seed=42, sort_metric='RMSE') # max models = num of models to try , seed = reporducibility , sort = sorts by rmse (lowest to highest)

    start = time.time() #time to train and select the best model
    aml.train(x=feature_cols, y=target_col, training_frame=train_h2o) #this trains the mode
    runtime = time.time() - start  # measures total time taken 

    best_name = aml.leader.algo #gets the best model
    print(f"\nH2O selected: {best_name}") #prints the name of the best model

    preds_h2o = aml.leader.predict(test_h2o) #makes predictions
    y_pred = preds_h2o.as_data_frame()['predict'].values #converts predictions back to numpy array
    mse, rmse, mae, precision, recall, f1 = evaluate_model(y_test, y_pred) #evaluates the model

    wrapped = H2OWrapper(aml.leader, feature_cols) #warps h20 model so that i can use the .predict fucntion
    prec_k, rec_k = show_top10( #this is calling my function to get top top 10 recommendations per user
        wrapped, preprocess, 'H2O AutoML', # wrapped is my trained h20 model, preprocesss is an object , "H20 AutoML" is a label
        X_test, y_test, sample_users_ids=[1, 2, 3])  #the features, the actual ratings and the users

    print_results('H2O AutoML', runtime, mse, rmse, mae,
                  precision, recall, f1, prec_k, rec_k) #prints the results 

    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime,
        'Precision@k': prec_k, 'Recall@k': rec_k,
        'Best Algorithm': best_name
    } #this returns a dictionary that stores all the results

  





class H2OWrapper:
    def __init__(self, leader, feature_cols):
        self.leader = leader  # stores the trained H2O model 
        self.feature_names_in_ = feature_cols  # saves the feature column names (needed so it matches training format)

    @staticmethod
    def _undummy(df):
        occ_cols = [c for c in df.columns if c.startswith("occupation_")]  # finds all one-hot encoded occupation columns
        if occ_cols:
            df = df.copy()  # makes a copy so we don’t change the original data

            # for each row it finds which occupation column has value 1
            df["occupation"] = (
                df[occ_cols]
                .idxmax(axis=1)  # gets the column name with the highest value (the 1)
                .str.replace("occupation_", "", regex=False)  # removes "occupation_" to keep just the job name
            )
            df = df.drop(columns=occ_cols)  # removes the dummy columns so we only have one occupation column
        return df  # returns updated dataframe

    def predict(self, X):
        df = self._undummy(X.copy())  # converts one-hot occupation columns back into a single column
        frame = h2o.H2OFrame(df)  # converts pandas dataframe into H2O format

        # if occupation column exists, convert it into categorical type
        if "occupation" in frame.columns:
            frame["occupation"] = frame["occupation"].asfactor()  # tells H2O this is a category, not a number

        preds = self.leader.predict(frame)  # uses the trained H2O model to make predictions

        return preds.as_data_frame()['predict'].values # converts predictions from H2O format back to numpy array 










#this function is what plots comparisons charts for the different models based on the evaluationmetrics to answer my research question
def save_comparison_chart(results, suffix= "automl"): #this stores the table as a dictionary with the model and its metrics
    
    model_names = list(results.keys()) #this extracts the model names
    colors = ["#052B4A", "#B40964", "#136C32"]#this gives each model a colour so i can distinuish between them

    #this displays name of the chart, the key (name in the results dicts), and the x-axis label
    metrics = [
        ("Precision", "Precision", "Which AutoML had top relevance?"),
        ("RMSE",      "RMSE",         "Which AutoML predicted ratings most accurately?"),
        ("Runtime",   "Runtime (s)",  "Which AutoML was fastest?"),
        ("Precision@k", "Precision@10", "Which AutoML gave most relevant top 10?"),
        ("Recall@k",    "Recall@10",  "Which AutoML found most liked movies in top 10?"),
    ]
    
    for key, ylabel, title in metrics: #for loop used to loop through each metric 
        
        vals = [results[m][key] for m in model_names] #stores the metric value for each model 

        plt.figure(figsize=(8, 5))  # makes a chart gets its own figure

        bars = plt.bar(model_names, vals, color=colors, edgecolor='white') #this draws a bar chart (one bar for each model)

        plt.title(title, fontsize=12, fontweight='bold')  # this adds a title to the chart
        plt.ylabel(ylabel, fontsize=10)                   # label for the y axis 
        plt.ylim(0, max(vals) * 1.25)                     # this is just so the y axis is taller then the highest value and doesnt get clipped 

        # loop through each bar and write its value on top of the bar just making it easier to analyse
        for bar, v in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # makes sure the text is at the centre of the bar
                bar.get_height() + 0.01,             # makes sure text is just above the bar
                f'{v:.4f}',                          # rounds the value to 4dp
                ha='center', va='bottom', fontsize=10 #this is just for alignment making it easier to read (aligning it horizontally and vertically)
            )

        plt.tight_layout()  # adjusts spacing so nothing gets removed because its too big

        # saves each chart as its own file 
        fname = f"chart_{key.lower()}_{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()  # close figure to free memory
        print(f"Saved: {fname}")









        
# this function is what will allow all the models to run in the correct order
def run():   
    preprocess = preprocessing() # creates an instance of the preprocessing class (so we can use its functions)

    X_train, X_test, y_train, y_test = preprocess.load_and_preprocess() # loads the dataset and splits into training + testing data

    print(f"Training set size: {X_train.shape[0]} rows") #shows how many rows are in training data
    print(f"Test set size:     {X_test.shape[0]} rows")  #shows how many rows are in test data                  
    print(f"Features:          {X_train.shape[1]} columns") #shows how many feature columns the model uses

    results = {} # dictionary to store results from each AutoML model

    try:
        # runs FLAML AutoML and stores results in dictionary under 'FLAML'
        results['FLAML'] = run_flaml(
            X_train, X_test, y_train, y_test,
            preprocess, time_budget=120) # time_budget controls how long FLAML searches for best model

    except Exception as e:
        print(f"\nFLAML failed: {e}") #if FLAML crashes, this prevents the whole program from stopping

    
    try:
        # runs PyCaret AutoML and stores results
        results['PyCaret'] = run_pycaret(
            X_train, X_test, y_train, y_test, preprocess)

    except Exception as e:
        print(f"\nPyCaret failed: {e}") # handles any errors so the next model can still run

   
    try:
        # runs H2O AutoML and stores results
        results['H2O AutoML'] = run_h2o(
            X_train, X_test, y_train, y_test,
            preprocess, max_models=10) # max_models limits how many models H2O tries

    except Exception as e:  
        print(f"\nH2O failed: {e}") #catches errors for H2O as well

    if not results:
        return #if all models failed, stop the function

    save_comparison_chart(results, suffix ="automl") # creates and saves comparison charts for all AutoML models

    # prints a clean summary table
    print("\n\n" + "-" * 70)
    print("  FINAL SUMMARY — AUTOML MODELS")
    print("-" * 70)

    # formats column headers so everything lines up nicely
    print(f"  {'Model':<20} {'MSE':>8}  {'RMSE':>8} {'MAE':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time':>8}")
    print("-" * 70) # line for separation

    # loops through each model and prints its results
    for name, m in results.items():
        print(
            f"  {name:<20} {m['MSE']:>8.4f} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
            f"{m['Precision']:>8.4f} {m['Recall']:>8.4f} "
            f"{m['F1']:>8.4f} {m['Runtime']:>7.1f}s"
        )

    print("-" * 70) # bottom line of the table 

    import json # imports json so we can save results to a file

    # saves all results into a JSON file 
    with open('results_automl.json', 'w') as f:
        json.dump(results, f)

    return results # returns results dictionary

run() # runs the whole pipeline




    

    
 
    
 
 
 
    
 
    


 
  
    
    
 
    
 
 
      
 
    


     
    


    



    
 
 
    



    
 


 




























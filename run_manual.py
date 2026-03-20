from recsys_framework.preprocessing import preprocessing #importing preprocessing class into my folder
from recsys_framework.manuel_models import ManualModels #importing manuel models class into my folder
from sklearn.metrics import mean_squared_error, mean_absolute_error #importing the mse and Rmse from the sklearn library
from sklearn.metrics import f1_score, recall_score, precision_score  #importing f1,precison and recall from the sklearn library
import numpy as np #importing numpy for mathematically operations
import time #importing time to work out the time each model takes to train
import matplotlib #load the matploy library
matplotlib.use("agg") #'agg' is needed for when i want to save files
import matplotlib.pyplot as plt #plt will be used to refer matplot library (for making graphs)
import shap  #import shap for explainability
import pandas as pd #import pandas for builidng dataframes




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
def print_results(model_name, runtime, mse, rmse, mae, precision, recall, f1):
  print("\n" + "-" *46) # this just creats a divider for readability 
  print(f"Model: {model_name}") #prints the name of the model shwoing the results
  print("-" * 46) #creates the repeat divider 

  print(f"Runtime:         {runtime: .2f} seconds")# print runtime to 2dp
  print(f" MSE:            {mse:.4f}") # print MSE to 4dp
  print(f" RMSE:           {rmse:.4f}") # print RMSE to 4dp
  print(f" MAE:            {mae:.4f}") #print MAE to 4dp
  print(f" Precision@10 :  {precision :.4f}") #print precison to 4dp
  print(f" Recall@10:      {recall:.4f}") #print recall  to 4dp
  print(f" F1 Score:       {f1:.4f}") #print f1 to 4dp
  print("-" * 46) #creates the repeat divider






#function to show the top 10 Recommendations per user
def show_top10(model, preprocess, model_name, sample_users_ids = None):
  print(f"\n  TOP 10 RECOMMENDATIONS  ({model_name})") #title
  if sample_users_ids is None:  #if no specific users are given, it will use the first 3 users in the dataset 
    sample_users_ids = preprocess.ratings["user_id"].unique()[:3] #this line gets all the unqiue users and takes the first 3

  for user_id in sample_users_ids: #loops though each sample user one at a time (so it will repeat 3 times since we chose 3)
    all_items = preprocess.items["item_id"].unique() # this line gets every movie id that exists in the dataset

    #the next code is what gets us a list of movies the user has already seen/rated
    rated_items = preprocess.ratings[ #looks inside ratings table
      preprocess.ratings["user_id"] == user_id #this keeps only rows for this user
    ]["item_id"].unique()# takes just the movie ids(unqiue removes duplicates)


    unrated_items = [i for i in all_items if i not in rated_items] #loops though all the movie ids and performs the operation as long as the movie is not already rated
    user_row = preprocess.users[preprocess.users["user_id"] == user_id] # making sure the user exists in the user dataset
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
        print(f"   {rank:2}. {title: <45} (predicted rating:{score:.1f}/5)") #this prints the numbered list (with the movie titles and the predicted rating)

    print("-" * 46) #line to seperate, so its ready clearly
  








#this function relates to explainability (so it explain why my model came up with the predictions it did)
def run_shap(model, X_test, model_name):
    print(" \n Running SHAP analysis ")
    X_sample = X_test.iloc[:500] #(takes only 500 rows from test data) Shap is slow so 500 is a compromise as 20,000 would take too long but 500 is quicker and its bug enough to learn from test 
    if model_name == 'Decision Tree':
        explainer = shap.TreeExplainer(model) #this uses the tree explainer for decison trees to calculate the SHAP values
        shap_values = explainer.shap_values(X_sample) # calculating SHAP
    else:
        background = shap.sample(X_sample, 50) #takes a smaller sample because this shap model is slower then the tree explainer
        explainer = shap.KernelExplainer(model.predict, background) #this is a general SHAP explainer for any model
        shap_values = explainer.shap_values(X_sample) #Calculating SHAP values


    # the next code is what will plot the SHAP impact bar chart 
    mean_shap = np.abs(shap_values).mean(axis=0) #np.abs removes negatives so directions dont cancel out and .mean(axis=0) averages across rows, giving one number per feature
    sorted_idx = np.argsort(mean_shap)[-15:] # get indices of the 15 most important features based on mean SHAP values

    #plotting the horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(
        [X_sample.columns[i] for i in sorted_idx],  # sets feature names as labels
        mean_shap[sorted_idx],  # these are their average impact scores
        color='#E91E63' # pink bars — just a different color to distinguish this chart from the feature importance chart
    )
    plt.xlabel("Mean |SHAP Value| — The average impact on predicted rating", fontsize=11) #this is the title of the chart
    plt.title(f"SHAP Feature Impact — {model_name}", fontsize=13, fontweight='bold') #this labels the model name on the chart
    plt.tight_layout() #adjusts spacing
    plt.savefig(f"shap_summary_{model_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight') # this saves the chart as a PNG file with the model name in the filename
    plt.close() #closes the figure to fill up memory
    print(f"  Saved: shap_summary_{model_name.replace(' ', '_')}.png") # this message just confirms the file was saved 








#this function is what plots comparisons charts for the different models based on the evaluationmetrics to answer my research question
def save_comparison_chart(results): #this stores the table as a dictionary with the model and its metrics
    
    model_names = list(results.keys()) #this extracts the model names
    colors = ['#2196F3', "#DC147F", "#3FFA80"]#this gives each model a colour so i can distinuish between them

    #this displays name of the chart, the key (name in the results dicts), and the x-axis label
    metrics = [
        ("Precision", "Precision@10", "Which model had best top-10 relevance?"),
        ("RMSE",      "RMSE",         "Which model predicted ratings most accurately?"),
        ("Runtime",   "Runtime (s)",  "Which model was fastest?")
    ]

    
    
    for key, ylabel, title, in metrics: #for loop used to loop through each metric 

        
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
        fname = f"chart_{key.lower()}_manual.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()  # close figure to free memory
        print(f"Saved: {fname}")






        
# this function is what will allow all the man models run in the correct order
def run():   
    print("\nLoading and preprocessing data...") # line that makes it clear we are loading and preprocessing
    preprocess = preprocessing() #creates instance of preprocess class
    Models = ManualModels() #creates an instance of manual models
    X_train, X_test, y_train, y_test = preprocess.load_and_preprocess() #performs preprocessing on data set

    print(f"Training set size: {X_train.shape[0]} rows")  # .shape[0]  checks the number of rows
    print(f"Test set size:     {X_test.shape[0]} rows")
    print(f"Features:          {X_train.shape[1]} columns")  # .shape[1] checks the number of columns

    results = {} #this stores all the metrics (so we can use it later in the comparison table )
  

    start = time.time() #this returns the current time to measure how long the model took
    dt_model = Models.decision_tree(X_train, y_train) #this trains the decsion tree model using the training data
    y_pred_dt = dt_model.predict(X_test) # this should return an array of prediction results
    
    runtime = time.time() - start #measures how long training and prediction took 
    mse, rmse, mae, precision, recall, f1 = evaluate_model(y_test, y_pred_dt) #this calculates all the metrics using the evaluate_model function by comparing the predictions to the real values
    
    #this stores the results that we have gotten inside the diction we made in the comparison table function
    results['Decision Tree'] = {
        'MSE': mse,'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime
    }
    print_results('Decision Tree', runtime, mse, rmse, mae, precision, recall, f1) #this prints the formated results for the decision tree model
    run_shap(dt_model, X_test, 'Decision Tree') #this prints the SHAP results for explainability
    show_top10(dt_model, preprocess, 'Decision Tree', sample_users_ids=[1, 2, 3]) #this shows the top 10 recommendations for 3 sample users


    start = time.time() #this is so runtime can be calculated it records the start time
    nb_model = Models.naive_bayes(X_train, y_train)#this trains the naive bayes model using the training data
    y_pred_nb = nb_model.predict(X_test)# this should return an array of prediction results
    runtime = time.time() - start #calculation of runtime
    
    #stores the results in the dictionary
    mse, rmse, mae, precision, recall, f1 = evaluate_model(y_test, y_pred_nb)
    results['Naive Bayes'] = { 'MSE': mse,
        'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime
    }
   #prints the results and top 10
    print_results('Naive Bayes', runtime, mse,  rmse, mae, precision, recall, f1)
    run_shap(nb_model, X_test,'Naive Bayes')
    show_top10(nb_model, preprocess, 'Naive Bayes', sample_users_ids=[1, 2, 3])


    start = time.time()
    mlp_model = Models.mlp(X_train, y_train)#this trains the mlp model using the training data
    y_pred_mlp = mlp_model.predict(X_test)# this should return an array of prediction results
    runtime = time.time() - start #calculates runtime

    #stores the results in the dictionary 
    mse, rmse, mae, precision, recall, f1 = evaluate_model(y_test, y_pred_mlp)
    results['MLP'] = { 'MSE': mse,
        'RMSE': rmse, 'MAE': mae,
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'Runtime': runtime
    }
    #prints results and top 10
    print_results('MLP Neural Network', runtime, mse, rmse, mae, precision, recall, f1)
    run_shap(mlp_model, X_test, 'MLP')
    show_top10(mlp_model, preprocess, 'MLP', sample_users_ids=[1, 2, 3])


    save_comparison_chart(results) #this saves all the results in the comparison table


    #this should print the comparison table of all 3 manual models
    print("\n\n" + "-" * 70)
    print("  FINAL SUMMARY — MANUAL MODELS")
    print("-" * 70)

    print(f"  {'Model':<20} {'MSE':>8}  {'RMSE':>8} {'MAE':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time':>8}") #this is used to line the columns up nicely so it can be read clearly
    print("-" * 70) #prints a line for seperation

    #this prints all the model results in one row each
    for name, m in results.items():
        print(
            f"  {name:<20} {m['MSE']:>8.4f} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
            f"{m['Precision']:>8.4f} {m['Recall']:>8.4f} "
            f"{m['F1']:>8.4f} {m['Runtime']:>7.1f}s"
        )
    print("-" * 70) #prints a line at the bottom of the table 
    return results

run()


    

    
 
    
 
 
 
    
 
    


 
  
    
    
 
    
 
 
      
 
    


     
    


    



    
 
 
    



    
 


 




























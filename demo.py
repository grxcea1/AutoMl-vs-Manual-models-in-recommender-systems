from recsys_framework.preprocessing import preprocessing # importing preprocessing function (handles loading and preparing data)
from recsys_framework.recommender import Recommender # importing recommender model (single model modes)
from recsys_framework.ensemble_recommender import ( 
    EnsembleRecommender) # importing ensemble recommender (combines FLAML + MLP)

# printing title for presentation
print("-" * 60)
print("  recsys_framework  —  Presentation Demo")
print("-" * 60)

# loading dataset
print("\nLoading MovieLens 100K...")
preprocess = preprocessing()  #creates a preprocessing object

# load and split data into train/test
X_train, X_test, y_train, y_test = (
    preprocess.load_and_preprocess())

# printing dataset info
print(f"  Train: {X_train.shape[0]} rows")  # number of training rows
print(f"  Test:  {X_test.shape[0]} rows")  # number of test rows
print(f"  Features: {X_train.shape[1]}")  # number of features

results = {}  #dictionary to store results for each model

# loop through different modes
for mode in ['accurate', 'fast', 'explainable']:
    
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")  #displays mode name
    print(f"{'='*60}")
    
    rec = Recommender(mode=mode)  #create recommender with selected mode
    
    rec.train(X_train, y_train)  #train model
    
    m = rec.evaluate(X_test, y_test)  #evaluate model
    
    results[mode] = m  #stores results
    
    # prints the key metrics
    print(f"  RMSE: {m['RMSE']:.4f}   "
          f"Runtime: {m['Runtime']:.1f}s")
    
    print(f"\n  Top 5 for User 1:")
    
    # generate top 5 recommendations
    for rank, (title, score) in enumerate(
            rec.recommend(user_id=1,
                          preprocess=preprocess,
                          k=5), start=1):
        
        print(f"    {rank}. {title:<45} "
              f"({score:.1f}/5)")  #prints rank, title and predicted rating


#ENSEMBLE MODEL
print(f"\n{'='*60}")
print("  MODE: ENSEMBLE (FLAML + MLP)")
print(f"{'='*60}")

ens = EnsembleRecommender()  #create ensemble model

ens.train(X_train, y_train)  #train ensemble

m = ens.evaluate(X_test, y_test)  #evaluate ensemble

results['ensemble'] = m  #store ensemble results

#prints metrics
print(f"  RMSE:      {m['RMSE']:.4f}")
print(f"  MAE:       {m['MAE']:.4f}")
print(f"  Precision: {m['Precision']:.4f}")
print(f"  Recall:    {m['Recall']:.4f}")
print(f"  F1:        {m['F1']:.4f}")
print(f"  Runtime:   {m['Runtime']:.1f}s")
print(f"\n  Top 5 for User 1:")

#generates top 5 recommendations using ensemble
for rank, (title, score) in enumerate(
        ens.recommend(user_id=1,
                      preprocess=preprocess,
                      k=5), start=1):
    
    print(f"    {rank}. {title:<45} "
          f"({score:.1f}/5)")


#SUMMARY SECTION
print(f"\n{'-'*60}")
print("  SUMMARY")
print(f"{'-'*60}")


#prints table header
print(f"  {'Mode':<14} {'RMSE':>8} "
      f"{'Runtime':>10} ")
print(f"  {'-'*60}")

#prints results for each model
for mode, m in results.items():
    print(f"  {mode:<14} {m['RMSE']:>8.4f} "
          f"{m['Runtime']:>8.1f}s   "
         )  #performance table

print("-" * 60)
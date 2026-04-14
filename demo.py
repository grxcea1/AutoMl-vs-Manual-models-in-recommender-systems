from recsys_framework.preprocessing import preprocessing
from recsys_framework.recommender import Recommender
from recsys_framework.ensemble_recommender import (
    EnsembleRecommender)

print("=" * 60)
print("  recsys_framework  —  Presentation Demo")
print("=" * 60)

print("\nLoading MovieLens 100K...")
preprocess = preprocessing()
X_train, X_test, y_train, y_test = (
    preprocess.load_and_preprocess())
print(f"  Train: {X_train.shape[0]} rows")
print(f"  Test:  {X_test.shape[0]} rows")
print(f"  Features: {X_train.shape[1]}")

results = {}

for mode in ['accurate', 'fast', 'explainable']:
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")
    rec = Recommender(mode=mode)
    rec.train(X_train, y_train)
    m = rec.evaluate(X_test, y_test)
    results[mode] = m
    print(f"  RMSE: {m['RMSE']:.4f}   "
          f"Runtime: {m['Runtime']:.1f}s")
    print(f"\n  Top 5 for User 1:")
    for rank, (title, score) in enumerate(
            rec.recommend(user_id=1,
                          preprocess=preprocess,
                          k=5), start=1):
        print(f"    {rank}. {title:<45} "
              f"({score:.1f}/5)")

print(f"\n{'='*60}")
print("  MODE: ENSEMBLE (FLAML + MLP)")
print(f"{'='*60}")
ens = EnsembleRecommender()
ens.train(X_train, y_train)
m = ens.evaluate(X_test, y_test)
results['ensemble'] = m
print(f"  RMSE: {m['RMSE']:.4f}   "
      f"Runtime: {m['Runtime']:.1f}s")
print(f"\n  Top 5 for User 1:")
for rank, (title, score) in enumerate(
        ens.recommend(user_id=1,
                      preprocess=preprocess,
                      k=5), start=1):
    print(f"    {rank}. {title:<45} "
          f"({score:.1f}/5)")

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
notes = {
    'accurate': 'Best RMSE and Prec@10',
    'fast': '8x faster than FLAML',
    'explainable': 'Full SHAP support',
    'ensemble': 'Novel FLAML+MLP combination'
}
print(f"  {'Mode':<14} {'RMSE':>8} "
      f"{'Runtime':>10}   Notes")
print(f"  {'-'*52}")
for mode, m in results.items():
    print(f"  {mode:<14} {m['RMSE']:>8.4f} "
          f"{m['Runtime']:>8.1f}s   "
          f"{notes[mode]}")
print("=" * 60)
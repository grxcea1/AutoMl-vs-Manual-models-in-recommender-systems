import numpy as np
import pandas as pd
import time
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              precision_score,
                              recall_score,
                              f1_score)


class Recommender:
    VALID_MODES = ('accurate', 'fast', 'explainable')

    def __init__(self, mode='accurate',
                 time_budget=120):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}")
        self.mode = mode
        self.time_budget = time_budget
        self.model = None
        self.feature_names_in_ = None
        self.trained = False
        self.runtime = None

    def train(self, X_train, y_train):
        self.feature_names_in_ = X_train.columns.tolist()
        start = time.time()

        if self.mode == 'accurate':
            from flaml import AutoML
            self.model = AutoML()
            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                task='regression',
                time_budget=self.time_budget,
                metric='rmse',
                verbose=0
            )
            print(f"  FLAML selected: "
                  f"{self.model.best_estimator}")

        elif self.mode == 'fast':
            from sklearn.neural_network import (
                MLPRegressor)
            self.model = MLPRegressor(
                activation='relu',
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
            self.model.fit(X_train, y_train)

        elif self.mode == 'explainable':
            from sklearn.tree import (
                DecisionTreeRegressor)
            self.model = DecisionTreeRegressor(
                random_state=42)
            self.model.fit(X_train, y_train)

        self.runtime = time.time() - start
        self.trained = True
        print(f"  [{self.mode}] trained in "
              f"{self.runtime:.1f}s")
        return self

    def predict(self, X):
        self._check_trained()
        X = self._align(X)
        return self.model.predict(X)

    def recommend(self, user_id, preprocess, k=10):
        self._check_trained()
        user_row = preprocess.users[
            preprocess.users['user_id'] == user_id]
        if user_row.empty:
            raise ValueError(
                f"user_id {user_id} not found.")

        rated = preprocess.ratings[
            preprocess.ratings['user_id'] == user_id
        ]['item_id'].unique()
        all_items = preprocess.items['item_id'].unique()
        candidates = [i for i in all_items
                      if i not in rated]
        if not candidates:
            return []

        df = pd.DataFrame({'item_id': candidates})
        df['user_id'] = user_id
        df = df.merge(preprocess.users,
                      on='user_id', how='left')
        df = df.merge(preprocess.items,
                      on='item_id', how='left')
        titles = df['title'].values

        drop_cols = ['timestamp', 'zip', 'title',
                     'release_date',
                     'video_release_date', 'IMDb_URL']
        drop_cols = [c for c in drop_cols
                     if c in df.columns]
        df = df.drop(columns=drop_cols).dropna()
        df['gender'] = df['gender'].map(
            {'M': 0, 'F': 1})
        df = pd.get_dummies(df, columns=['occupation'])
        df = self._align(df)

        if df.empty:
            return []

        preds = self.model.predict(df)
        top_idx = np.argsort(preds)[::-1][:k]
        return [(titles[i], float(preds[i]))
                for i in top_idx if i < len(titles)]

    def evaluate(self, X_test, y_test):
        self._check_trained()
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        y_bin_t = (np.array(y_test) >= 4).astype(int)
        y_bin_p = (np.array(y_pred) >= 4).astype(int)
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

    def _align(self, X):
        if self.feature_names_in_ is None:
            return X
        X = X.copy()
        for col in self.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_names_in_]

    def _check_trained(self):
        if not self.trained:
            raise RuntimeError(
                "Call train() first.")
from recsys_framework.preprocessing import preprocessing
from recsys_framework.manuel_models import ManualModels
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, recall_score, precision_score 
import numpy as np


def run():
  
  # instance of preprocessing class
  preprocess = preprocessing()

  # instance of ManualModels class
  Models = ManualModels()

  # splitting data
  X_train, X_test, y_train, y_test = preprocess.load_and_preprocess()

  # Training the Model (decison tree)
  decision_tree = Models.decision_tree(X_train, y_train)

  # getting results from fitted model using test data
  y_model = decision_tree.predict(X_test)

  # evaluation
  mse = mean_squared_error(y_test, y_model)  
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, y_model)

  # binarize for classification metrics
  y_test_binary = (y_test >= 4).astype(int)
  y_model_binary = (y_model >= 4).astype(int)

  precision = precision_score(y_test_binary, y_model_binary)
  recall = recall_score(y_test_binary, y_model_binary)
  f1 = f1_score(y_test_binary, y_model_binary)

  return rmse, mae, precision, recall, f1



rmse, mae, precision, recall, f1 = run()
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
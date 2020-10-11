import numpy as np

def preprocess_for_tabnet(X,y):
  X = X.reset_index()
  y = y.reset_index()
  y = y['granted_number_of_nights']
  return np.array(X), np.array(y)
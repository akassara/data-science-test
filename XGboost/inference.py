import time
import pandas as pd
from Preprocessing.XGBpreproc import preprocess
import numpy as np
from sklearn.metrics import log_loss
import xgboost as xgb



#import data
requests_test = pd.read_csv('data/requests_test.csv')
#Data preprocessing
# Dataframe of categorical variables:
categorical_val= list(requests_test.select_dtypes(include=[np.object]))
categorical_val.remove('request_id')
#we separe between categorical variables and date variables to preprocess it separately
categorical_val.remove('answer_creation_date')
categorical_val.remove('group_creation_date')
categorical_val.remove('request_creation_date')
categorical_val.remove('victim_of_violence_type')
date_columns = ['answer_creation_date','group_creation_date','request_creation_date']
X_test , y_test = preprocess(requests_test,categorical_val,date_columns)
# Define the test scorer
def competition_scorer(y_true, y_pred):
    return log_loss(y_true, y_pred, sample_weight=10**y_true)

#retrieve saved model
best_model = xgb.Booster()
PATH = '/data-science-test/model_zoo/xgb_model.model'
best_model.load_model(PATH)
print('The model has been loaded')
#run inference
start = time.time()
preds = best_model.predict(xgb.DMatrix(X_test))
end = time.time()
score = competition_scorer(y_test, preds)
print('time per prediction:' ,(end-start)/len(X_test))
print('The competition score on test data', score)
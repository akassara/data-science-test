import pandas as pd
import numpy as np
from Preprocessing.XGBpreproc import preprocess
from Preprocessing.TABNETpreproc import preprocess_for_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
from XGboost.inference import competition_scorer
import time


#import data
requests_test = pd.read_csv('../data/requests_test.csv')
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

# Drop Nan value because otherwise there are memory errors
X_test['granted_number_of_nights'] = y_test
X_test = X_test.dropna()
y_test = X_test['granted_number_of_nights']
X_test = X_test.drop(columns = ['granted_number_of_nights'])
#preprocess the datasets for TabNet
X_test_tab, y_test_tab = preprocess_for_tabnet(X_test,y_test)
# retrieve model
# Not working with version <3.7

PATH = '../model_zoo/TabNet_model.zip'
clf = TabNetClassifier()
clf.load_model(PATH)

#run inference
start = time.time()
preds = clf.predict_proba(X_test_tab)
end = time.time()
score = competition_scorer(y_test_tab, preds)
print('time per prediction:' ,(end-start)/len(X_test))
print('The competition score on test data', score)
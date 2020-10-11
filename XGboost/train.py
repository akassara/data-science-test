import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from Preprocessing.XGBpreproc import preprocess


# import data to dataframe
requests_train = pd.read_csv('data/requests_train.csv')
# Dataframe of categorical variables:
categorical_val= list(requests_train.select_dtypes(include=[np.object]))
categorical_val.remove('request_id')
#we separe between categorical variables and date variables to preprocess it separately
categorical_val.remove('answer_creation_date')
categorical_val.remove('group_creation_date')
categorical_val.remove('request_creation_date')
categorical_val.remove('victim_of_violence_type')
date_columns = ['answer_creation_date','group_creation_date','request_creation_date']
# We transform the dataframe into encoded features and target
X,y = preprocess(requests_train,categorical_val,date_columns)

# split between the train and the validation samples
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

xgb_model = xgb.XGBClassifier(max_depth = 4,learning_rate = 0.01,n_estimators=10000)
xgb_model.fit(X_train, y_train, sample_weight=10**y_train, eval_set=[(X_val, y_val)],eval_metric = 'mlogloss',early_stopping_rounds= 100,sample_weight_eval_set=[10**y_val])
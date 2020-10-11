from sklearn.model_selection import train_test_split
from Preprocessing.TABNETpreproc import preprocess_for_tabnet
from Preprocessing.XGBpreproc import preprocess
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import numpy as np
import torch




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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, y_train = preprocess_for_tabnet(X_train,y_train)
X_val , y_val = preprocess_for_tabnet(X_val ,y_val)
clf =  TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=1e-1))
clf.device= 'cuda'
weights = {0:1,1:10,2:10**2,3:10**3}
clf.fit(X_train=X_train, y_train=y_train, ##Train features and train targets
                X_valid=X_val, y_valid=y_val, ##Valid features and valid targets
                weights=weights,
                max_epochs=5,##Maxiµmum number of epochs during training
                patience=2, ##Number of consecutive non improving epoch before early stopping
                batch_size=16, ##Training batch size
                )
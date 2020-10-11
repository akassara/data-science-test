from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_victims_of_violence_type(df):
  no_violence_victim_df = df.loc[df['victim_of_violence']=='f']
  violence_victim_df = df.loc[df['victim_of_violence']=='t']
  df['victim_of_violence_type'] = df['victim_of_violence_type'].astype(str)
  encoder = LabelEncoder()
  df['encoded_victim_of_violence_type'] = encoder.fit_transform(df['victim_of_violence_type'])
  for i in no_violence_victim_df.index:
    df['encoded_victim_of_violence_type'][i] = -1
  for i in range(len(df)):
    df['encoded_victim_of_violence_type'][i] += 1
  df = df.drop(columns=['victim_of_violence_type'])
  return df


def encode_categorical_variables(df,columns):
  for column in columns:
    df[column] = df[column].astype(str)
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    df.loc[df[column].isnull(),column] = -1
  return df

def encode_date_variable(df,columns):
  for column in columns:
    df[column] = pd.to_datetime(df[column])
    df[column+'_year'] = df[column].dt.year
    df[column+'_month'] = df[column].dt.month
    df = df.drop(columns = [column])
  return df

def preprocess(df,cat_columns,date_columns):
  """ Transform dataframe into to encoded features and targets
  """
  df = encode_victims_of_violence_type(df)
  df = encode_categorical_variables(df,cat_columns)
  df = encode_date_variable(df,date_columns)
  features = list(df.columns)
  features.remove('request_id')
  features.remove('granted_number_of_nights')
  X = df[features]
  y = df["granted_number_of_nights"]
  return X,y
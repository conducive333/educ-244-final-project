import models.model
import pandas as pd
import numpy as np
import joblib
import os

class Cluster3Model(models.model.Model):

  DIR = os.path.join(models.model.Model.ASSETS, 'cluster3')

  def __init__(self):
    super().__init__('cluster3', np.array([32.4969853, -84.45705188]))
    self.model = joblib.load(os.path.join(Cluster3Model.DIR, 'model.joblib'))
    self.backup = joblib.load(os.path.join(Cluster3Model.DIR, 'backup.joblib'))
    self.baseline = joblib.load(os.path.join(Cluster3Model.DIR, 'baseline.joblib'))
    self.preprocessor = joblib.load(os.path.join(Cluster3Model.DIR, 'preprocessor.joblib'))

  def transform(self, datapoint: pd.DataFrame) -> any:
    df = datapoint.copy()

    '''Create a column of pets_allowed'''
    temp = df[['cats_allowed', 'dogs_allowed']].apply(sum, axis=1).item()
    if temp == 0:
      df['pets_allowed'] = 0
    else:
      df['pets_allowed'] = 1

    '''Create a park_dummy'''
    park_2 = ['attached garage']
    park_0 = ['off-street parking']
    if df['parking_options'].item() in park_2:
      df['park_dummy'] = 'two'
    elif df['parking_options'].item() in park_0:
      df['park_dummy'] = 'zero'
    else:
      df['park_dummy'] = 'one'

    '''Create a laundry_dummy'''
    in_unit = ['w/d in unit']
    no_laundry = ['no laundry on site']
    if df['laundry_options'].item() in in_unit:
      df['laundry_dummy'] = 'in_unit'
    elif df['laundry_options'].item() in no_laundry:
      df['laundry_dummy'] = 'no_laundry'
    else:
      df['laundry_dummy'] = 'in_building'

    '''Convert boolean columns into zero-one'''
    bool_cols = ['smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'comes_furnished']
    for bool_col in bool_cols:
      df[bool_col] = df[bool_col].astype(int)
  
    '''Select only columns for modeling'''
    df_processed = df.copy()[
      [
        'type', 
        'pets_allowed', 
        'smoking_allowed', 
        'wheelchair_access', 
        'electric_vehicle_charge',
        'comes_furnished', 
        'laundry_dummy', 
        'park_dummy', 
        'sqfeet', 
        'beds', 
        'baths', 
        'lat', 
        'long', 
        'state'
      ]
    ]

    return self.preprocessor.transform(df_processed).toarray()

  def predict(self, datapoint: pd.DataFrame, neighbors: pd.DataFrame) -> float:
    try:
      return self.model.predict(self.transform(datapoint))
    except Exception:
      print('Model failed! Using backup. ', end='')
      try:
        return self.backup[datapoint['type'][0]][datapoint['state'][0]].predict(datapoint[['lat', 'long']])
      except KeyError:
        print('Backup failed! Using baseline.')
        return self.baseline.predict(datapoint[['lat', 'long']])
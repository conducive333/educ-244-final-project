import models.model
import pandas as pd
import numpy as np
import joblib
import os

class Cluster0Model(models.model.Model):

  DIR = os.path.join(models.model.Model.ASSETS, 'cluster0')

  def __init__(self):
    super().__init__('cluster0', np.array([37.24252129, -97.30641693]))
    self.model = joblib.load(os.path.join(Cluster0Model.DIR, 'model.joblib'))
    self.backup = joblib.load(os.path.join(Cluster0Model.DIR, 'backup.joblib'))
    self.baseline = joblib.load(os.path.join(Cluster0Model.DIR, 'baseline.joblib'))
    self.preprocessor = joblib.load(os.path.join(Cluster0Model.DIR, 'preprocessor.joblib'))

  def transform(self, datapoint: pd.DataFrame) -> any:
    bool_cols = ['cats_allowed','dogs_allowed','smoking_allowed','wheelchair_access','electric_vehicle_charge','comes_furnished','comes_furnished']
    datapoint = datapoint.copy()
    datapoint['sqfeet']=np.log(datapoint['sqfeet'])
    datapoint.loc[datapoint['parking_options'].isnull(),'parking_options'] = 'unknown'
    datapoint.loc[datapoint['laundry_options'].isnull(),'laundry_options'] = 'unknown'
    for bool_col in bool_cols:
      datapoint[bool_col] = datapoint[bool_col].astype(int)
    return self.preprocessor.transform(datapoint).toarray()

  def predict(self, datapoint: pd.DataFrame, neighbors: pd.DataFrame) -> float:
    try:
      unit_price_predict = self.model.predict(self.transform(datapoint))
      return float(unit_price_predict * datapoint['sqfeet'][-1])
    except Exception:
      print('Model failed! Using backup. ', end='')
      try:
        return self.backup[datapoint['type'][0]][datapoint['state'][0]].predict(datapoint[['lat', 'long']])
      except KeyError:
        print('Backup failed! Using baseline.')
        return self.baseline.predict(datapoint[['lat', 'long']])
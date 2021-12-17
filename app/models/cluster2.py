import models.model
import pandas as pd
import numpy as np
import joblib
import os

class Cluster2Model(models.model.Model):

  DIR = os.path.join(models.model.Model.ASSETS, 'cluster2')

  def __init__(self):
    super().__init__('cluster2', np.array([39.55708038, -119.80023203]))
    self.model = joblib.load(os.path.join(Cluster2Model.DIR, 'model.joblib'))
    self.baseline = joblib.load(os.path.join(Cluster2Model.DIR, 'baseline.joblib'))

  def transform(self, datapoint: pd.DataFrame) -> any:
    raise NotImplementedError()

  def predict(self, datapoint: pd.DataFrame, neighbors: pd.DataFrame) -> float:
    try:
      modl = self.model[datapoint['type'][0]][datapoint['state'][0]]
      return modl.predict(datapoint[['lat', 'long']])
    except KeyError:
      return self.baseline.predict(datapoint[['lat', 'long']])
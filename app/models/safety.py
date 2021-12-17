import models.model
import pandas as pd
import numpy as np
import joblib
import os 

class SafetyModel(models.model.Model):

  DIR = os.path.join(models.model.Model.ASSETS, 'utils')

  def __init__(self):
    super().__init__('kdtree', np.array([0, 0]))
    self.model = joblib.load(os.path.join(SafetyModel.DIR, 'safety.joblib'))

  def transform(self, datapoint: pd.DataFrame) -> any:
    raise NotImplementedError()

  def predict(self, datapoint: pd.DataFrame) -> float:
    return self.model.predict(datapoint[['lat', 'long']])

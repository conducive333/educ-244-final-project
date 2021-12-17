import pandas as pd
import numpy as np
import abc
import os

class Model(abc.ABC): 

  ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models')

  def __init__(self, name: str, centroid: np.array):
    self.centroid = centroid
    self.name = name

  @abc.abstractmethod
  def transform(self, datapoint: pd.DataFrame) -> any:
    """
    Applies a transformation to the input datapoint so that the
    model can make a prediction using it.
    
    Parameters:
    -----------
    
      datapoint : pd.DataFrame
        A data frame with the following keys and types:
          'lat': [float], # example [1.0]
          'long': [float], # example [2.0]
          'beds': [float], # example [1.0]
          'baths': [float], # example [2.0]
          'sqfeet': [float], # example [1000.0]
          'state': [str], # example ['ca']
          'type': [str], # example ['apartment']
          'laundry_options': [str], # example ['nan']
          'parking_options': [str], # example ['nan']
          'cats_allowed': [bool], # example [True] 
          'dogs_allowed': [bool], # example [False]
          'smoking_allowed': [bool], # example [False]
          'wheelchair_access': [bool], # example [True] 
          'electric_vehicle_charge': [bool], # example [False] 
          'comes_furnished': [bool], # example [True]
    
    Returns:
    --------

      point : Any
        A transformed datapoint.

    """
    pass

  @abc.abstractmethod
  def predict(self, datapoint: pd.DataFrame, neighbors: pd.DataFrame) -> float:
    """
    Returns a prediction for the input datapoint.
    
    Parameters:
    -----------
    
      datapoint : pd.DataFrame
        A data frame with the following keys and types:
          'lat': [float], # example [1.0]
          'long': [float], # example [2.0]
          'beds': [float], # example [1.0]
          'baths': [float], # example [2.0]
          'sqfeet': [float], # example [1000.0]
          'state': [str], # example ['ca']
          'type': [str], # example ['apartment']
          'laundry_options': [str], # example ['nan']
          'parking_options': [str], # example ['nan']
          'cats_allowed': [bool], # example [True] 
          'dogs_allowed': [bool], # example [False]
          'smoking_allowed': [bool], # example [False]
          'wheelchair_access': [bool], # example [True] 
          'electric_vehicle_charge': [bool], # example [False] 
          'comes_furnished': [bool], # example [True]
    
    Returns:
    --------

      price : float
        A price prediction given the input datapoint.

    """
    pass

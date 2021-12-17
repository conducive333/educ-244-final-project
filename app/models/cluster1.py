import sklearn.preprocessing as skprep
import sklearn.impute as skimp
import models.model
import pandas as pd
import numpy as np
import joblib
import os

class Cluster1Model(models.model.Model):

  DIR = os.path.join(models.model.Model.ASSETS, 'cluster1')

  # Obtained from data analysis
  MIN_PRICE = 622.0000047009901
  MAX_PRICE = 1674.766488318389

  def __init__(self):
    super().__init__('cluster1', np.array([40.18590738, -78.9847621]))
    self.models = joblib.load(os.path.join(Cluster1Model.DIR, 'model.joblib'))
    self.backup = joblib.load(os.path.join(Cluster1Model.DIR, 'backup.joblib'))
    self.baseline = joblib.load(os.path.join(Cluster1Model.DIR, 'baseline.joblib'))
    self.preprocessor = joblib.load(os.path.join(Cluster1Model.DIR, 'preprocessor.joblib'))
    self.capitals = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'assets', 'data', 'us-state-capitals.csv.gz'), compression='gzip')

  def transform(self, datapoint: pd.DataFrame) -> any:
    transformed = self.one_hot_encode_bools(datapoint)
    transformed = self.add_distance_to_capitals(transformed)
    transformed = self.add_neighborhood_distances(transformed)
    transformed = self.add_landmark_distances(transformed)
    transformed = self.encode_categorical(transformed)
    transformed = transformed.drop(columns=[
      'state',
      'type',
      'long',
      'lat'
    ])
    return self.normalize(transformed)

  def predict(self, datapoint: pd.DataFrame, neighbors: pd.DataFrame) -> float:
    try:
      if Cluster1Model.MIN_PRICE <= neighbors['price'].mean() <= Cluster1Model.MAX_PRICE:
        predictions = pd.DataFrame()
        transformed = self.transform(datapoint)
        for i, model in enumerate(self.models):
          m = model['estimator']
          predictions[f'model{i}'] = m.predict(transformed[m.feature_names_in_])
        return predictions.mean(axis=1)
      else:
        raise Exception()
    except Exception:
      print('Model failed! Using backup. ', end='')
      try:
        return self.backup[datapoint['type'][0]][datapoint['state'][0]].predict(datapoint[['lat', 'long']])
      except KeyError:
        print('Backup failed! Using baseline.')
        return self.baseline.predict(datapoint[['lat', 'long']])

  def encode_categorical(self, df):
    df['parking_options'] = self.preprocessor['po_labeler'].transform(df['parking_options'])
    df['laundry_options'] = self.preprocessor['lo_labeler'].transform(df['laundry_options'])
    return df

  def one_hot_encode_bools(self, df):
    df['is_fancy'] = (df['electric_vehicle_charge'] | df['comes_furnished']).astype(int)
    df['pets_allowed'] = (df['cats_allowed'] | df['dogs_allowed']).astype(int)
    df['wheelchair_access'] = df['wheelchair_access'].astype(int)
    df['smoking_allowed'] = df['smoking_allowed'].astype(int)
    df = df.drop(columns=[
      'electric_vehicle_charge', 
      'comes_furnished',
      'cats_allowed', 
      'dogs_allowed', 
    ])
    return df

  def add_landmark_distances(self, df):
    for landmarks in landmark_mtx:
      for name, coord in landmarks.items():
        df[f'{name}_dist'] = np.linalg.norm(df[['lat', 'long']] - coord, axis=1)
    return df

  def add_neighborhood_distances(self, df):
    for neighborhoods in neighborhood_mtx:
      for name, coord in neighborhoods.items():
        df[f'{name}_dist'] = np.linalg.norm(df[['lat', 'long']] - coord, axis=1)
    return df

  def add_distance_to_capitals(self, df):
    for idx, row in self.capitals.iterrows():
      name = row['name'].lower()
      coord = np.array(row[['latitude', 'longitude']])
      diff = df[['lat', 'long']] - coord       
      df[f'{name}_captital_dist'] = np.sqrt(np.sum(diff**2, axis=1)) # np.linalg.norm(df[['lat', 'long']] - coord, axis=1)
    return df

  def normalize(self, df):
    normalizer = self.preprocessor['normalizer']
    feat_names = normalizer.feature_names_in_
    df[feat_names] = normalizer.transform(df[feat_names])
    return df

# Michigan neighborhoods
mi_neighborhoods = {
  'okemos': np.array([45.87297786197644, -84.62847041655759]),
  'troy': np.array([42.60703761607627, -83.14736420368314]),
  'bloomfield_charter_township': np.array([42.578869359094526, -83.28152819512678]),
}

# North Carolina neighborhoods
nc_neighborhoods = {
  'winston_salem': np.array([36.101161790823475, -80.25788567632397]),
  'dilworth': np.array([35.2059586709318, -80.8526019294176]),
  'ballantyne_east': np.array([35.04932445555813, -80.83701659903802]),
}

# Ohio neighborhoods
oh_neighborhoods = {
  'oakwood': np.array([39.72550586216518, -84.17403651393656]),
  'ottawa_hills': np.array([41.663907814157966, -83.6423491073066]),
  'shaker_heights': np.array([41.47422244557466, -81.53450528913191]),
}

# Virginia neighborhoods
va_neighborhoods = {
  'alexandria': np.array([38.821497134409825, -77.08342063624801]),
  'arlington': np.array([38.88133010512791, -77.10765120666845]),
  'virginia_square': np.array([38.886090947281, -77.10565021044826]),
}

# Pennsylvania neighborhoods
pa_neighborhoods = {
  'chesnut_hill': np.array([40.070817050879945, -75.20659474905997]),
  'society_hill': np.array([39.94327154623738, -75.14706922091864]),
  'center_city': np.array([39.95069948727436, -75.15726612901973])
}

# New York neighborhoods
ny_neighborhoods = {
  'harlem': np.array([40.818716685807196, -73.9531526828147]),
  'east_upper_side': np.array([40.773453174735195, -73.95610333952337]),
  'williamsburg': np.array([40.70810551330598, -73.95737732613472])
}

# Maryland neighborhoods
md_neighborhoods = {
  'inner_harbor': np.array([39.28599533214056, -76.61360387628632]),
  'fells_point': np.array([39.28419298758618, -76.59348398990699]),
  'federal_hill_montgomery': np.array([39.27903248553645, -76.61134679969817]),
}

# New Jersey neighborhoods
nj_neighborhoods = {
  'princeton': np.array([40.35771107138268, -74.67327763403718]),
  'montclair': np.array([40.82784665757987, -74.21081163134934]),
  'madison': np.array([40.759353821095374, -74.4182147267064])
}

# Massachusetts neighborhoods
ma_neighborhoods = {
  'waltham': np.array([42.38619739627199, -71.24153429507528]),
  'brookline': np.array([42.32764059193201, -71.13960029625804]),
  'lexington': np.array([42.4457795839414, -71.23195695733767])
}

# Indiana neighborhoods
in_neighborhoods = {
  'carmel': np.array([39.97113646533522, -86.12827196900389]),
  'fishers': np.array([39.9602564608419, -85.97185162365706]),
  'munster': np.array([41.54891387796956, -87.50400611266197])
}

neighborhood_mtx = [
  mi_neighborhoods,
  nc_neighborhoods,
  oh_neighborhoods,

  # The Virginia neighborhoods weren't very useful features!
  # va_neighborhoods,

  pa_neighborhoods,
  ny_neighborhoods,
  md_neighborhoods,
  nj_neighborhoods,
  ma_neighborhoods,
  in_neighborhoods,
]

# Michigan landmarks
mi_landmarks = {
  'mackinac_island': np.array([45.87297786197644, -84.62847041655759]),
  'detroit_inst_of_arts': np.array([42.35948234171309, -83.06444735757302]),
  'the_henry_ford': np.array([42.305822586490294, -83.22477837333459]),
}

# North Carolina landmarks
nc_landmarks = {
  'clingmans_dome': np.array([35.57007350100404, -83.49824381623084]),
  'biltmore': np.array([35.6889928232918, -82.57080484838305]),
  'pisgah_national_forest': np.array([35.35065365942371, -82.74035998738512]),
}

# Ohio landmarks
oh_landmarks = {
  'cedar_point': np.array([41.48238380688474, -82.68350987316123]),
  'cuyahoga_valley_national_park': np.array([41.27966778629776, -81.5652779343297]),
  'cleveland_museum_of_art': np.array([41.50916962794644, -81.6120488443251]),
}

# Virginia landmarks   
va_landmarks = {
  'virginia_beach': np.array([36.8514053000573, -75.98244867063534]),
  'colonial_williamsburg': np.array([37.27898483726433, -76.69885566531059]),
  'george_washingtons_mt_vernon': np.array([38.70810756212071, -77.08619675973303]),
}

# Pennsylvania landmarks
pa_landmarks = {
  'philadelphia_museum_of_art': np.array([39.96569302292908, -75.180987559704]),
  'hersheypark': np.array([40.286555267364605, -76.65048005846148]),
  'presque_isle_state_park': np.array([42.15852155871631, -80.11506692685606])
}

# New York landmarks
ny_landmarks = {
  'empire_state_building': np.array([40.748626458963656, -73.9857635020144]),
  'central_park': np.array([40.78118337440834, -73.96651379585668]),
  'museum_of_modern_art': np.array([40.76157082786343, -73.9775894155076])
}

# Stores the center of the U.S.A. Coordinates are given as (lat, lon)
mainland = {
  'mainland': np.array([39.828175, -98.5795])
}

# Let's see if housing units near these landmarks are hot spots!
landmark_mtx = [
  mi_landmarks,
  nc_landmarks,
  oh_landmarks,
  
  # All these landmarks had the lowest correlation with price! Why Virginia? Lol.
  # va_landmarks,

  pa_landmarks,
  ny_landmarks,
  mainland
]
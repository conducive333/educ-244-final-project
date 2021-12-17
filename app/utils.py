import reverse_geocoder as rg
import models.safety as safe
import models.cluster0 as c0
import models.cluster1 as c1
import models.cluster2 as c2
import models.cluster3 as c3
import plotly.express as px
import models.model as mdl
import pandas as pd
import numpy as np
import joblib
import os

# Get all pre-trained models
safety = safe.SafetyModel()
models = [
  c0.Cluster0Model(), # JY
  c1.Cluster1Model(), # CD
  c2.Cluster2Model(),
  c3.Cluster3Model(), # KM
]

# Load dataset
# TODO: Heroku deployment fails bc this is eating too much memory
# You'll either need to use a database or read the file in chunks 
# and replace some of the df[' ... '].unique() calls with pre-filled
# values. NOTE: deploy with `git push heroku main`.
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'assets', 'data', 'all_states.csv.gz'), compression='gzip')

# Loads a K-D tree pre-trained on the 'all_states.csv' dataset.
# This model can quickly identify nearest neighbors for a given
# input point, which leads to faster map re-renders.
kdt = joblib.load(os.path.join(os.path.dirname(__file__), 'assets', 'models', 'utils', 'kdtree.joblib'))

def get_data():
  return df

def get_state(lat: float, lon: float) -> str:
  return state_mapping.get(rg.search((lat, lon), mode=1)[0]['admin1'], None)

def get_neighbors(lat: float, lon: float, k: int) -> list[int]:
  indices = kdt.query([[lat, lon]], k=k, return_distance=False)[0]
  return df.iloc[indices]

def are_valid(*args: list[any]) -> bool:
  return all(map(lambda elem: elem is not None, args))

def resolve_model(lat: float, lon: float) -> mdl.Model:
  return min(models, key=lambda m: np.linalg.norm(np.array([lat, lon]) - m.centroid))

def predict(datapoint: pd.DataFrame) -> pd.DataFrame:
  try:
    lat, lon = datapoint['lat'][0], datapoint['long'][0]
    model = resolve_model(lat, lon)
    ngbrs = get_neighbors(lat, lon, k=30)
    print(f'Using "{model.name}". ', end='')
    datapoint['price'] = model.predict(datapoint.copy(), ngbrs)
    return datapoint.append(ngbrs)
  except Exception:
    return safety.predict(datapoint)

def create_map(data: pd.DataFrame, lat: float, lon: float, zoom: float = 2.5) -> any:
  return px.scatter_mapbox(
    data
    , lat='lat'
    , lon='long'
    , color='price'
    , size='sqfeet'
    , color_continuous_scale=px.colors.sequential.Magma
    , mapbox_style='open-street-map' # Doesn't require mapbox API token!
    , size_max=15
    , zoom=zoom # lower values = less zoom
    , height=500
    , center={
      'lat': lat,
      'lon': lon
    }
  )

state_mapping = {
  'Alabama': 'al',
  'Alaska': 'ak',
  'Arizona': 'az',
  'Arkansas': 'ar',
  'California': 'ca',
  'Colorado': 'co',
  'Connecticut': 'ct',
  'Delaware': 'de',
  'Hawaii': 'hi',
  'Florida': 'fl',
  'Georgia': 'ga',
  'Idaho': 'id',
  'Illinois': 'il',
  'Indiana': 'in',
  'Iowa': 'ia',
  'Kansas': 'ks',
  'Kentucky': 'ky',
  'Louisiana': 'la',
  'Maine': 'me',
  'Maryland': 'md',
  'Massachusetts': 'ma',
  'Michigan': 'mi',
  'Minnesota': 'mn',
  'Mississippi': 'ms',
  'Missouri': 'mo',
  'Montana': 'mt',
  'Nebraska': 'ne',
  'Nevada': 'nv',
  'New Hampshire': 'nh',
  'New Jersey': 'nj',
  'New Mexico': 'nm',
  'North Carolina': 'nc',
  'North Dakota': 'nd',
  'New York': 'ny',
  'Ohio': 'oh',
  'Oklahoma': 'ok',
  'Oregon': 'or',
  'Pennsylvania': 'pa',
  'Rhode Island': 'ri',
  'South Carolina': 'sc',
  'South Dakota': 'sd',
  'Tennessee': 'tn',
  'Texas': 'tx',
  'Utah': 'ut',
  'Vermont': 'vt',
  'Virginia': 'va',
  'Washington': 'wa',
  'West Virginia': 'wv',
  'Wisconsin': 'wi',
  'Wyoming': 'wy'
}
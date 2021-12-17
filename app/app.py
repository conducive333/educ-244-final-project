import dash_bootstrap_components as dbc
import dash.html as html
import dash.dcc as dcc
import pandas as pd
import utils
import dash
import os

# Create the dashboard app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# DO NOT DELETE THIS - this is required for Heroku deployment to succeed
server = app.server

# Center of the U.S.
DEFAULT_LAT = 39.828175
DEFAULT_LON = -98.5795

# Get mappings for input forms
df = utils.get_data()
laundry_map = { str(i) : label for i, label in enumerate(df['laundry_options'].unique(), 1) }
parking_map = { str(i) : label for i, label in enumerate(df['parking_options'].unique(), 1) }
region_map = { str(i) : label for i, label in enumerate(df['region'].str.lower().unique(), 1) }
state_map = { str(i) : label for i, label in enumerate(df['state'].str.upper().unique(), 1) }
htype_map = { str(i) : label for i, label in enumerate(df['type'].str.lower().unique(), 1) }
checklist = {
  'cats_allowed': 'Cats Allowed',
  'dogs_allowed': 'Dogs Allowed',
  'smoking_allowed': 'Smoking Allowed',
  'wheelchair_access': 'Wheelchair Access',
  'electric_vehicle_charge': 'Electric Vehicle Charge',
  'comes_furnished': 'Comes Furnished'
}

# Skeleton for app
app.layout = html.Div([
  dbc.NavbarSimple(brand='Welcome!', color='secondary', dark=True),
  dbc.Row([
    dbc.Col([
      dbc.Container([
        dcc.Graph(id='map')
      ])
    ])
  ]),
  dbc.Form([
    dbc.Row([
      dbc.Col([
        dbc.FormFloating([
          dbc.Input(id='latitude', placeholder='latitude', type='number', value=DEFAULT_LAT),
          dbc.Label('latitude'),
        ]),
      ], width=2, style={'padding': '0px 20px 0px 20px'}),
      dbc.Col([
        dbc.FormFloating([
          dbc.Input(id='longitude', placeholder='longitude', type='number', value=DEFAULT_LON),
          dbc.Label('longitude'),
        ]),
      ], width=2, style={'padding': '0px 20px 0px 20px'}),
      dbc.Col([
        dbc.FormFloating([
          dbc.Input(id='beds', placeholder='beds', type='number', min=0, value=2),
          dbc.Label('beds'),
        ]),
      ], width=2, style={'padding': '0px 20px 0px 20px'}),
      dbc.Col([
        dbc.FormFloating([
          dbc.Input(id='baths', placeholder='baths', type='number', min=0, value=2),
          dbc.Label('baths'),
        ]),
      ], width=2, style={'padding': '0px 20px 0px 20px'}),
      dbc.Col([
        dbc.FormFloating([
          dbc.Input(id='sqfeet', placeholder='sqfeet', type='number', min=0, value=1000),
          dbc.Label('sqfeet'),
        ]),
      ], width=3, style={'padding': '0px 20px 0px 20px'}),
      dbc.Col([
        dbc.InputGroup([
          dbc.InputGroupText('laundry options'),
          dbc.Select(id='laundry', value='1', options=[{ 'label': label, 'value': value } for value, label in laundry_map.items()]),
        ]),
        dbc.InputGroup([
          dbc.InputGroupText('parking options'),
          dbc.Select(id='parking', value='1', options=[{ 'label': label, 'value': value } for value, label in parking_map.items()]),
        ]),
        dbc.InputGroup([
          dbc.InputGroupText('housing type'),
          dbc.Select(id='htype', value='1', options=[{ 'label': label, 'value': value } for value, label in htype_map.items()]),
        ]),
      ], style={'padding': '20px 20px 20px 20px'}),
      dbc.Col([
        dbc.Checklist(
          id='checklist', 
          value=list(checklist.keys()),
          options=[{ 'label': label, 'value': value } for value, label in checklist.items()
        ], style={'padding': '20px 0px 0px 0px'}),
      ], width=3, style={'padding': '0px 30px 0px 20px'}),
    ], className='mb-3'),
  ]),
])

@app.callback(
  dash.dependencies.Output('map', 'figure'),
  dash.dependencies.Input('latitude', 'value'),
  dash.dependencies.Input('longitude', 'value'),
  dash.dependencies.Input('beds', 'value'),
  dash.dependencies.Input('baths', 'value'),
  dash.dependencies.Input('sqfeet', 'value'),
  dash.dependencies.Input('htype', 'value'),
  dash.dependencies.Input('laundry', 'value'),
  dash.dependencies.Input('parking', 'value'),
  dash.dependencies.Input('checklist', 'value'),
)
def render_map(latitude, longitude, beds, baths, sqfeet, htype, laundry, parking, checklist):
  if (utils.are_valid(latitude, longitude, beds, baths, sqfeet, htype, laundry, parking, checklist)):
    checklist_set = set(checklist)
    prediction = utils.predict(
      pd.DataFrame({
        'electric_vehicle_charge': ['electric_vehicle_charge' in checklist_set], 
        'wheelchair_access': ['wheelchair_access' in checklist_set], 
        'smoking_allowed': ['smoking_allowed' in checklist_set],
        'comes_furnished': ['comes_furnished' in checklist_set],
        'cats_allowed': ['cats_allowed' in checklist_set], 
        'dogs_allowed': ['dogs_allowed' in checklist_set],
        'laundry_options': [laundry_map[laundry]],
        'parking_options': [parking_map[parking]],
        'state': [utils.get_state(latitude, longitude)],
        'type': [htype_map[htype]],
        'long': [longitude],
        'sqfeet': [sqfeet],
        'lat': [latitude],
        'baths': [baths],
        'beds': [beds],
      })
    )
    return utils.create_map(prediction, latitude, longitude, zoom=5)
  else:
    return utils.create_map(df.iloc[:0], DEFAULT_LAT, DEFAULT_LON)

if __name__ == '__main__':
  if os.environ.get('PY_ENV') == 'production':
    from waitress import serve
    serve(app.server, host="0.0.0.0", port=3000)
  else:
    app.run_server(port=3000, debug=True)

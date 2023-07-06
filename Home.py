import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from plotly.subplots import make_subplots
import geopandas as gpd
import folium 
from streamlit_folium import st_folium
import numpy as np
from streamlit_folium import folium_static

import os


st.set_page_config(layout="wide")

st.title("GeoData Visualization Platform")





# load dataframe and information
def load_data(path):
    df = gpd.read_file(path)
    return df

# load geojson
#with open('zoned_data_sit.geojson') as f:
#    geo= json.load(f)



def threshold(data):
    threshold_scale = np.linspace(data.min(), data.max(), 6, dtype=float)
    threshold_scale[-1] = float(threshold_scale[-1]) + 1
    return threshold_scale.tolist()


def show_map(data,data_geo,threshold_scale,variable):
    maps = folium.Choropleth(
        geo_data=data_geo,
        data=data,
        columns=['zone',variable],
        threshold_scale=threshold_scale,
        key_on='feature.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name=variable
    ).add_to(stream_map)

    folium.LayerControl().add_to(stream_map)
    tooltip = folium.features.GeoJsonTooltip(fields=['zone', variable],
                                             aliases=['Zone', 'Mean concentration'],
                                             labels=True)
    maps.geojson.add_child(tooltip)
    
    folium_static(stream_map)


map_dict = {'Concentrations map': 'data/zoned_data_sit.geojson', 'Chemicals map': 'data/zoned_chem_sit.geojson'}

select_map = st.sidebar.selectbox('Select map', ['Concentrations map', 'Chemicals map'])

df = load_data(map_dict[select_map])

if select_map == 'Concentrations map':
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry', 'zone'])]))
    df = df[[select_variable,'zone', 'geometry']]

if select_map == 'Chemicals map':
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry', 'zone'])]))
    df = df[[select_variable,'zone', 'geometry']]



stream_map = folium.Map(location=[6.2518400, -75.5635900], zoom_start=10, control_scale=True)
show_map(df,df,threshold(df[select_variable]),select_variable)

st.sidebar.title("About")

logo = "logo-eafit.png"
st.sidebar.image(logo)


# cito - exposición - supervivencia


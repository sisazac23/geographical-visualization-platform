import pandas as pd
import streamlit as st
import geopandas as gpd
import folium 
import numpy as np
from streamlit_folium import folium_static

import os


st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("GeoData Visualization Platform")

# load dataframe and information
def load_data(path):
    df = gpd.read_file(path)
    return df

def threshold(data):
    threshold_scale = np.linspace(data.min(), data.max(), 6, dtype=float)
    #threshold_scale[-1] = float(threshold_scale[-1]) + 1
    return threshold_scale.tolist()

def show_map(data, data_geo, threshold_scale, variable):
    maps = folium.Choropleth(
        geo_data=data_geo,
        data=data,
        columns=['zone', variable],
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


def plot_points(data, size=10000, color='red'):
    stream_map = folium.Map(location=[6.2518400, -75.5635900], zoom_start=10, control_scale=True, tiles=select_tile_provider,locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
    for i in range(len(data)):
        location = [data.iloc[i].geometry.y, data.iloc[i].geometry.x]
        folium.CircleMarker(location=location, radius=5, color=color, fill=True, fill_color=color).add_to(stream_map)

    folium.LayerControl().add_to(stream_map)
    
    folium_static(stream_map)




tile_providers = {
    'OpenStreetMap': 'OpenStreetMap',
    'Stamen Terrain': 'Stamen Terrain',
    'Stamen Toner': 'Stamen Toner',
    'CartoDB Positron': 'CartoDB Positron',
    'CartoDB Dark_Matter': 'CartoDB Dark_Matter',
}

map_dict = {'Concentrations map': 'data/zoned_data_sit.geojson', 'Chemicals map': 'data/zoned_chem_sit.geojson',
            'Metal risk map': 'data/zoned_metalic_sit.geojson','Mission map 1':'data/mission1.geojson','Mission map 2':'data/mission2.geojson'}


select_tile_provider = st.sidebar.selectbox('Select tile provider', list(tile_providers.keys()))
select_map = st.sidebar.selectbox('Select map', ['Concentrations map', 'Chemicals map', 'Metal risk map', 'Missions'])


if select_map == 'Missions':
    select_mission = st.sidebar.selectbox('Select mission', ['Mission map 1', 'Mission map 2'])
    df = load_data(map_dict[select_mission])
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry','lat','lot'])]))
    df = df[[select_variable, 'geometry','lat','lot']]

if select_map == 'Concentrations map':
    df = load_data(map_dict[select_map])
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry', 'zone'])]))
    df = df[[select_variable, 'zone', 'geometry']]

if select_map == 'Chemicals map':
    df = load_data(map_dict[select_map])
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry', 'zone'])]))
    df = df[[select_variable, 'zone', 'geometry']]

if select_map == 'Metal risk map':
    df = load_data(map_dict[select_map])
    select_variable = 'risk'
    df = df[[select_variable,'geometry','zone']]

if select_map != 'Missions':
    stream_map = folium.Map(location=[6.2518400, -75.5635900], zoom_start=10, control_scale=True, tiles=select_tile_provider,locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
    show_map(df, df, threshold(df[select_variable]), select_variable)
else:
    plot_points(df)
    
    st.markdown(f"## {select_variable} Line Chart")
    # beautify the st.line_chart(df[select_variable], use_container_width=True) plot
    st.line_chart(df[select_variable], use_container_width=True)
    st.markdown(f"## {select_variable} Histogram")
    st.bar_chart(df[select_variable], use_container_width=True)
    # center the descriptive statistics markdown st.markdown(f"## {select_variable} Descriptive Statistics")
    st.markdown(f"## {select_variable} Descriptive Statistics")
    st.table(df[select_variable].describe().round(2).transpose())


       
st.sidebar.title("About")
markdown = "Platform to visualize chemical measurements over different colombian points of interest" 
st.sidebar.info(markdown)
logo = "logo-eafit.png"
st.sidebar.image(logo)



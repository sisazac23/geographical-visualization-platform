import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import folium 
import numpy as np
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
from streamlit_folium import folium_static
import plotly.graph_objects as go


import os
import io


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

def save_geojson_with_bytesio(dataframe):
    #Function to return bytesIO of the geojson
    shp = io.BytesIO()
    dataframe.to_file(shp,  driver='GeoJSON')
    return shp

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def threshold(data):
    threshold_scale = np.linspace(data.min(), data.max(), 6, dtype=float)
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
    figs = []
    st.markdown(f"## {select_variable} Line Chart")
    # beautify the st.line_chart(df[select_variable], use_container_width=True) plot
    st.line_chart(df[select_variable], use_container_width=True)
    # plot using plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[select_variable], mode='lines+markers'))
    figs.append(fig)
    st.markdown(f"## {select_variable} Histogram")
    # plot using plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[select_variable]))
    figs.append(fig)
    st.bar_chart(df[select_variable], use_container_width=True)
    # center the descriptive statistics markdown st.markdown(f"## {select_variable} Descriptive Statistics")
    st.markdown(f"## {select_variable} Descriptive Statistics")
    # create fig using plotly
    st.table(df[select_variable].describe().round(2).transpose())
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[list(df[select_variable].describe().round(2).index),
                           list(df[select_variable].describe().round(3).values)],
                   fill_color='lavender',
                   align='left'))
    ])
    figs.append(fig)



if select_map != 'Missions':
    st.download_button(
        label="Download data",
        data=save_geojson_with_bytesio(df),
        file_name=str(select_map)+str(select_variable)+'.geojson',
        mime='application/geo+json',
    )
else:
    st.download_button(
        label="Download data",
        data=save_geojson_with_bytesio(df),
        file_name=str(select_map)+str(select_mission)+str(select_variable)+'.geojson',
        mime='application/geo+json',
    )

if select_map == 'Missions':
    export_as_pdf = st.button("Export Report")
    if export_as_pdf:
        pdf = FPDF()
        for fig in figs:
            pdf.add_page()
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.write_image(tmpfile.name)
                    pdf.image(tmpfile.name, 10, 10, 200, 100)
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Report_"+str(select_mission)+'_'+str(select_variable))
        st.markdown(html, unsafe_allow_html=True)
       
st.sidebar.title("About")
markdown = "Platform to visualize chemical measurements over different colombian points of interest" 
st.sidebar.info(markdown)
logo = "logo-eafit.png"
st.sidebar.image(logo)


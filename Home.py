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

import glob
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


def create_3d_grid(dataframe: pd.DataFrame, resolution: int):
    """
    Creates a 3D grid from a Pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to create the grid from.
        resolution (int): The number of points to use in each dimension.
    
    Returns:
        np.ndarray, np.ndarray, np.ndarray: The 3D grid.
    """
    
    # Get the minimum and maximum values for each column
    min_values = dataframe.min(numeric_only=True)
    max_values = dataframe.max(numeric_only=True)

    # Create coordinate arrays using the minimum and maximum values
    x = np.linspace(min_values[0], max_values[0], resolution)
    y = np.linspace(min_values[1], max_values[1], resolution)
    z = np.linspace(min_values[2], max_values[2], resolution)

    # Create a 3D grid using NumPy's meshgrid function
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    # Return the 3D grid
    return x_grid, y_grid, z_grid

def plot_grid_and_points(df,x, y, z,variable):
    """
    Plot the cells formed by the meshgrid.

    Args:
        x (ndarray): X-coordinates of the meshgrid.
        y (ndarray): Y-coordinates of the meshgrid.
        z (ndarray): Z-coordinates of the meshgrid.
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    

    # Get the dimensions of the grid
    x_dim, y_dim, z_dim = x.shape

    # Plot the lines between the cells
    for i in range(x_dim - 1):
        for j in range(y_dim - 1):
            for k in range(z_dim - 1):
                # Get the vertices of the current cell
                vertices = [
                    (x[i, j, k], y[i, j, k], z[i, j, k]),
                    (x[i+1, j, k], y[i+1, j, k], z[i+1, j, k]),
                    (x[i+1, j+1, k], y[i+1, j+1, k], z[i+1, j+1, k]),
                    (x[i, j+1, k], y[i, j+1, k], z[i, j+1, k]),
                    (x[i, j, k+1], y[i, j, k+1], z[i, j, k+1]),
                    (x[i+1, j, k+1], y[i+1, j, k+1], z[i+1, j, k+1]),
                    (x[i+1, j+1, k+1], y[i+1, j+1, k+1], z[i+1, j+1, k+1]),
                    (x[i, j+1, k+1], y[i, j+1, k+1], z[i, j+1, k+1])
                ]

                # Define the edges of the current cell
                edges = [
                    (vertices[0], vertices[1]),
                    (vertices[1], vertices[2]),
                    (vertices[2], vertices[3]),
                    (vertices[3], vertices[0]),
                    (vertices[4], vertices[5]),
                    (vertices[5], vertices[6]),
                    (vertices[6], vertices[7]),
                    (vertices[7], vertices[4]),
                    (vertices[0], vertices[4]),
                    (vertices[1], vertices[5]),
                    (vertices[2], vertices[6]),
                    (vertices[3], vertices[7])
                ]

                # Plot the lines for the edges
                for edge in edges:
                    x_vals = [edge[0][0], edge[1][0]]
                    y_vals = [edge[0][1], edge[1][1]]
                    z_vals = [edge[0][2], edge[1][2]]
                    ax.plot(x_vals, y_vals, z_vals, 'r-',alpha=0.2)
                    
    ax.scatter(df['lat'],df['lot'],df['A_1'],c=df[variable],cmap='viridis',alpha=0.5)

    ax.set_xlabel('lat')
    ax.set_ylabel('lon')
    ax.set_zlabel('altitude')

    return ax


def plot_cells(x, y, z):
    """
    Plot the cells formed by the meshgrid.

    Args:
        x (ndarray): X-coordinates of the meshgrid.
        y (ndarray): Y-coordinates of the meshgrid.
        z (ndarray): Z-coordinates of the meshgrid.
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    

    # Get the dimensions of the grid
    x_dim, y_dim, z_dim = x.shape

    # Plot the lines between the cells
    for i in range(x_dim - 1):
        for j in range(y_dim - 1):
            for k in range(z_dim - 1):
                # Get the vertices of the current cell
                vertices = [
                    (x[i, j, k], y[i, j, k], z[i, j, k]),
                    (x[i+1, j, k], y[i+1, j, k], z[i+1, j, k]),
                    (x[i+1, j+1, k], y[i+1, j+1, k], z[i+1, j+1, k]),
                    (x[i, j+1, k], y[i, j+1, k], z[i, j+1, k]),
                    (x[i, j, k+1], y[i, j, k+1], z[i, j, k+1]),
                    (x[i+1, j, k+1], y[i+1, j, k+1], z[i+1, j, k+1]),
                    (x[i+1, j+1, k+1], y[i+1, j+1, k+1], z[i+1, j+1, k+1]),
                    (x[i, j+1, k+1], y[i, j+1, k+1], z[i, j+1, k+1])
                ]

                # Define the edges of the current cell
                edges = [
                    (vertices[0], vertices[1]),
                    (vertices[1], vertices[2]),
                    (vertices[2], vertices[3]),
                    (vertices[3], vertices[0]),
                    (vertices[4], vertices[5]),
                    (vertices[5], vertices[6]),
                    (vertices[6], vertices[7]),
                    (vertices[7], vertices[4]),
                    (vertices[0], vertices[4]),
                    (vertices[1], vertices[5]),
                    (vertices[2], vertices[6]),
                    (vertices[3], vertices[7])
                ]

                # Plot the lines for the edges
                for edge in edges:
                    x_vals = [edge[0][0], edge[1][0]]
                    y_vals = [edge[0][1], edge[1][1]]
                    z_vals = [edge[0][2], edge[1][2]]
                    ax.plot(x_vals, y_vals, z_vals, 'r-',alpha=0.2)

    ax.set_xlabel('lat')
    ax.set_ylabel('lot')
    ax.set_zlabel('A_1')

    return ax

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
            'Metal risk map': 'data/zoned_metalic_sit.geojson'}

path = 'data/geojson_missions'
all_files = glob.glob(path + "/*.geojson" )
mission_names = []
for filename in all_files:
    mission_names.append(filename.split('/')[-1].split('.')[0])
    map_dict[filename.split('/')[-1].split('.')[0]]=path+'/'+filename.split('/')[-1]

select_tile_provider = st.sidebar.selectbox('Select tile provider', list(tile_providers.keys()))
select_map = st.sidebar.selectbox('Select map', ['Concentrations map', 'Chemicals map', 'Metal risk map', 'Missions'])

# write title of the map
st.markdown(f"## {select_map}")

if select_map == 'Missions':
    select_mission = st.sidebar.selectbox('Select mission', mission_names)
    df = load_data(map_dict[select_mission])
    select_variable = st.sidebar.selectbox('Select variable', list(df.columns[~df.columns.isin(['geometry','lat','lot'])]))
    #df = df[[select_variable, 'geometry','lat','lot']]

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
    # Add a scatter trace with lines and markers
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[select_variable],
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),  # Set line color and width
        marker=dict(symbol='circle', size=8, color='#3498db', line=dict(color='#ffffff', width=1)),  # Set marker properties
    ))

    # Update layout for better appearance
    fig.update_layout(
        title='Line Plot with Markers of {}'.format(select_variable),
        xaxis_title='Index',
        yaxis_title=select_variable,
        showlegend=False,  # Hide legend
        plot_bgcolor='white',  # Set plot background color
        xaxis=dict(
            showline=True,
            showgrid=False,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor='lightgray'
        )
    )
    figs.append(fig)
    st.markdown(f"## {select_variable} Histogram")
    # plot using plotly
    fig = go.Figure()
    # Add a histogram trace
    fig.add_trace(go.Histogram(
        x=df[select_variable],
        marker_color='#3498db',  # Set the color of the bars
        opacity=0.7  # Set the opacity of the bars
    ))
    # Update layout for better appearance
    fig.update_layout(
        title='Histogram of {}'.format(select_variable),
        xaxis_title=select_variable,
        yaxis_title='Frequency',
        showlegend=False,  # Hide legend
        bargap=0.05,  # Gap between bars
        plot_bgcolor='white',  # Set plot background color
        xaxis=dict(
            showline=True,
            showgrid=False,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor='lightgray'
        )
    )
    figs.append(fig)
    st.bar_chart(df[select_variable], use_container_width=True)
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
    x_grid, y_grid, z_grid = create_3d_grid(df, 5)
    ax = plot_grid_and_points(df, x_grid, y_grid, z_grid,select_variable)
    st.pyplot(ax.figure)
    



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


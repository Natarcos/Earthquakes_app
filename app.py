#---------------------------------- Libraries--------------------------------- 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# -----------------------------------------------------------------------------


#---------------------------------- Page Config---------------------------------
st.set_page_config(
    page_title="Earthquake Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------------------------------------------------------------


#---------------------------------- Title and Description----------------------
st.title("Earthquake Analysis Dashboard")
st.markdown("""
    This dashboard provides an analysis of earthquake data using various visualizations and clustering techniques.
    The data is sourced from the USGS Earthquake Catalog and includes information on earthquake magnitude, depth, and location.
    The dashboard allows users to explore the data through interactive charts and maps, as well as perform clustering analysis.
""")
# -----------------------------------------------------------------------------


#---------------------------------- Vars in None------------------------------------
# The objetive of this section is to define the variables that will be used in the app.
filtered_df = None
df = None
# -----------------------------------------------------------------------------


#---------------------------------- Load Data-----------------------------------
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv('data/all_month.csv')
        # Convert time column to datetime and arase the time zone
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        # aditionals columns
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        # magnitude categories
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Micro', 'Light', 'Moderate', 'Strong']
        df['magnitude_category'] = np.select(conditions, choices, default='Unknown')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
#---------------------------------------------------------------------------------------


#---------------------------------- day of week transformation-----------------------------------
days_translation = {
    'Monday': 'Mon',
    'Tuesday': 'Tue',
    'Wednesday': 'Wed',
    'Thursday': 'Thu',
    'Friday': 'Fri',
    'Saturday': 'Sat',
    'Sunday': 'Sun'
}
#---------------------------------------------------------------------------------------


#---------------------------------- Color Scheme-----------------------------------
magnitude_colors = {
    'Micro': 'blue',  # Blue
    'Light': 'green',  # Green
    'Moderate': 'orange',  # Orange
    'Strong': 'red'  # Red
}
#---------------------------------------------------------------------------------------


#---------------------------------- Function to control positives sizes in markers-----------------------------------
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)
#---------------------------------------------------------------------------------------


#---------------------------------- Data Loading-----------------------------------
try:
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is not None and not df.empty:
        st.sidebar.header("Data Filters") #data filters
        min_date = df['time'].min().date() 
        max_date = df['time'].max().date()
        
        #filter
        data_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
        #transform the date range to datetime
        if len(data_range) == 2:
            start_date,  end_date = data_range
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)].copy()

    else:
        filtered_df = df.copy()

#MAG FILTER
    min_mag , max_mag = st.sidebar.slider(
        "Select magnitude range",
        min_value=float(df['mag'].min()),
        max_value=float(df['mag'].max()),
        value=(float(df['mag'].min()), float(df['mag'].max())),
        step=0.1
    )
    filtered_df = filtered_df[(filtered_df['mag'] >= min_mag) & (filtered_df['mag'] <= max_mag)].copy()

#DEPTH FILTER
    min_depth, max_depth = st.sidebar.slider(
        "Select depth range (km)",
        min_value=float(df['depth'].min()),
        max_value=float(df['depth'].max()),
        value=(float(df['depth'].min()), float(df['depth'].max())),
        step=5.0
    )
    filtered_df = filtered_df[(filtered_df['depth'] >= min_depth) & (filtered_df['depth'] <= max_depth)].copy()


#EVENT FILTER
    event_types = df['type'].unique().tolist()
    selected_types = st.sidebar.multiselect(
        "Select event types",
        options=event_types,
        default=event_types
    )

#REGION FILTER (TODO: CHECK SPAIN DATA AND REPAIR THE RFILTER)
    all_regions = sorted(df['place'].str.split(', ').str[-1].unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Filtrar por región",
        options=all_regions,
        default=[]
    )
    if selected_regions:
        region_mask = filtered_df['place'].str.contains('|'.join(selected_regions), case=False)
        filtered_df = filtered_df[region_mask]

#COUNT EVENTS FILTERS
    st.sidebar.metric('Selected Events', len(filtered_df))


#ADVANCE OPTIONS 
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Options")

    show_cluster = st.sidebar.checkbox("Show Clustering", value=False)
    show_advanced_charts = st.sidebar.checkbox("Show Advanced Charts", value=False)

    # Verify if the filtered_df is not empty
    if len(filtered_df) == 0:
        st.warning("No data available for the selected filters.")
    else:
        #principal tabs to show the data
        main_tabs = st.tabs(["📊 General Summary", "🌐 Geographic Analysis", "⏱️ Temporal Analysis", "📈 Advanced Analysis"])
        with main_tabs[0]:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Events", len(filtered_df))
            col2.metric("Average Magnitude", f"{filtered_df['mag'].mean():.2f}")
            col3.metric("Maximum Magnitude", f"{filtered_df['mag'].max():.2f}")
            col4.metric("Average Depth", f"{filtered_df['depth'].mean():.2f} km")

            #mag distribution and depth distribution
            col_dist1, col_dist2 = st.columns(2)

            with col_dist1:
                st.subheader("Magnitude Distribution")
                
                fig_mag = px.histogram(
                    filtered_df,
                    x="mag",
                    nbins=30,
                    color="magnitude_category",
                    color_discrete_map=magnitude_colors,
                    labels={"mag": "Magnitude", "count": "Frequency"},
                    title="Magnitude Distribution by Category"
                )
                fig_mag.update_layout(bargap=0.1)
                st.plotly_chart(fig_mag, use_container_width=True)
            
            with col_dist2:
                st.subheader("Depth Distribution")
                
                fig_depth = px.histogram(
                    filtered_df,
                    x="depth",
                    nbins=30,
                    color="magnitude_category",
                    color_discrete_map=magnitude_colors,
                    labels={"depth": "Depth (km)", "count": "Frequency"},
                    title="Depth Distribution by Magnitude Category"
                )
                fig_depth.update_layout(bargap=0.1)
                st.plotly_chart(fig_depth, use_container_width=True)

            st.subheader("Relation between Magnitude and Depth")
            size_values = ensure_positive(filtered_df['mag'])

            fig_scatter = px.scatter(
                filtered_df,
                x="depth",
                y="mag",
                color="magnitude_category",
                size=size_values,
                size_max=15,
                opacity=0.7,
                hover_name="place",
                hover_data=["time", "updated"],
                labels={"depth": "Depth (km)", "mag": "Magnitude"},
            )

            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)


            #top10 places
            st.subheader("Top 10 Places with Most Events")
            top_places = filtered_df['place'].value_counts().head(10).reset_index()
            top_places.columns = ['Place', 'Count']

            fig_top = px.bar(
                top_places,
                x='Count',
                y='Place',
                orientation='h',
                color='Count',
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Top 10 Places with Most Events",
                labels={"Count": "Number of Events", "Place": "Place"}
            )
            fig_top.update_traces(text_position='outside')
            fig_top.update_layout(yaxis = {'categoryorder':'total ascending'}, height=400)
            st.plotly_chart(fig_top, use_container_width=True)

            # Tab 2: Geographic Analysis
            with main_tabs[1]:
                geo_tabs = st.tabs(["Event Map", "Heat Map", "Cluster Analysis"])
                
                # Tab 1: Event Map
                with geo_tabs[0]:
                    st.subheader("Geographic Distribution of Earthquakes")
                    
                    # Create a basic map with px.scatter_geo instead of scatter_map
                    fig_map = px.scatter_geo(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        color="magnitude_category",
                        size=ensure_positive(filtered_df['mag']),  # Ensure positive values
                        size_max=15,
                        hover_name="place",
                        hover_data={
                            "latitude": False,
                            "longitude": False,
                            "magnitude_category": False,
                            "mag": ":.2f",
                            "depth": ":.2f km",
                            "time": True,
                            "type": True
                        },
                        color_discrete_map=magnitude_colors,
                        projection="natural earth"
                    )
                    
                    fig_map.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600,
                        geo=dict(
                            showland=True,
                            landcolor="lightgray",
                            showocean=True,
                            oceancolor="lightblue",
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white"
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # List of significant events
                    st.subheader("Significant Events (Magnitude ≥ 4.0)")
                    significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not significant_events.empty:
                        st.dataframe(
                            significant_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No events with magnitude ≥ 4.0 in the selected range.")
                
                # Tab 2: Heat Map
                with geo_tabs[1]:
                    st.subheader("Heat Map of Seismic Activity")
                    st.markdown("""
                    This heat map shows areas with higher concentration of seismic activity.
                    Brighter areas indicate higher density of events.
                    """)
                    
                    # Use a heatmap approach with scatter_geo and markersize for the heat map
                    fig_heat = px.density_mapbox(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        z=ensure_positive(filtered_df['mag']),  # Ensure that z is positive
                        radius=10,
                        center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                        zoom=1,
                        mapbox_style="open-street-map",
                        opacity=0.8
                    )
                    
                    fig_heat.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Show significant events as a table instead of additional map
                    st.subheader("Significant Events (Magnitude ≥ 4.0)")
                    strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not strong_events.empty:
                        st.dataframe(
                            strong_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No events with magnitude ≥ 4.0 in the selected range.")
    #---------------------------------------------------------------------------------------
            









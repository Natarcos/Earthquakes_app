#---------------------------Librerias-------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#---------------------------------------------------------------

#-----------------------------Page Config-----------------------
st.set_page_config(
    page_title = "Earthquake Analysis"
    page_icon= "📊",
    layotu="wide",
    initial_sidebar_state="expanded"
)
#-----------------------------------------------------------------

#---------------------------Title and Description-----------------
st.title("Earthquake Analysis Dashboard")
st.markdown("""
    This dashboard provides an analysis of earthquake data using various visualizations and clustering techniques.
    The data is sourced from the USGS Earthquake Catalog and includes information on earthquake magnitude, depth, and location.
    The dashboard allows users to explore the data through interactive charts and maps, as well as perform clustering analysis.
""")
#------------------------------------------------------------------

#------------------------------Var in None-------------------------

#The objective of this section is to define the variables that will be used in the app.
filtered_df = None 
df = None
#-----------------------------------------------------------------

#--------------------------Load data------------------------------
@st.cache_data(ttl=3600)
def load_data()
    try:
        df= pd.read_csv('data/all_month.csv')
        #convert time columns to datetime and erase the time zone
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        #aditionals columns
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        # magnitude categories
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >=2.0) & (df['mag'] < 4.0),
            (df['mag'] >=4.0) & (df['mag'] < 6.0),
            (df['mag'] < 6.0)
        ]
        choices = ['Micro', 'Light', 'Moderate', 'Strong']
        df['magnitude_category'] = np.select(conditions,choices,default='Unknown')
        return df
    except Exception as e:
        st.error(f'Error loading data:{e}')
        return None
#-----------------------------------------------------------------

#--------------------------day of week transformation------------------------------
days_translation = {
    'Monday': 'Mon',
    'Tuesday': 'True',
    'Wednesday': 'Wed',
    'Thursday': 'Thu',
    'Friday': 'Fri',
    'Saturday': 'Sat',
    'Sunday': 'Sun'
}
#-----------------------------------------------------------------

#--------------------------Color Scheme------------------------------
magnitude_colors = {
    'Micro': 'blue',  # Blue
    'Light': 'green',  # Green
    'Moderate': 'orange',  # Orange
    'Strong': 'red'  # Red
}
#-----------------------------------------------------------------

#--------------------------Function to control positives sizes in markers------------------------------
def ensure_positive (values, min=3)
    if isinstance(values, (pd.Series, np.ndarray, lists)):
        return np.maximum(np-abs(values), min_size)
    else:
        return max(abs(values), min_size)
#-----------------------------------------------------------------

#--------------------------Data loading-------------------------------
Try:
    with st.spinner("Loading data..."):
    df = load_data()
    if df is not None and not df.empty:
        st.sidebar.header("Data Filters")
        min_date = df['time'].min().date()
        max_date = df['time'].max().date()
        
        # Date filter
        data_range = st.sidebar.date_input("Select date range",[min_date, max_date],min_value=min_date,max_value=max_date)
        #transform the date range to datetime
        if len(data_range) == 2:
            start_date, end_date = data_range
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd. Timedelta(days=1) - pd.Timedelta(seconds=1)
            
        filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)].copy()
        
    else:
        filtered_df = df.copy()
#-----------------------------------------------------------------

#MAG FILTER
    min_mag, max_mag = st.sidebar.slider()
#-----------------------------------------------------------------


        




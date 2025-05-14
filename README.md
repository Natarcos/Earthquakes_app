# Seismic Activity Dashboard

This interactive dashboard provides a comprehensive analysis of seismic activity using data visualization and statistical techniques. Built with Streamlit and Plotly Express, it allows users to explore earthquake data, identify patterns, and gain insights into seismic events. 

https://earthquakesapp-natalia-arcos.streamlit.app/

![Seismic Activity Dashboard](Image/output_20250512_182829.jpg)

## Features

-   **Data Loading and Filtering:**
    -   Loads seismic data from a CSV file (`all_month.csv`).
    -   Filters data by date range, magnitude, depth, event type, and region.
-   **General Summary:**
    -   Displays key metrics such as total events, average magnitude, maximum magnitude, and average depth.
    -   Visualizes magnitude and depth distributions using histograms.
    -   Shows the relationship between magnitude and depth using scatter plots.
    -   Identifies the top 10 regions with the highest seismic activity using bar charts.
-   **Geographic Analysis:**
    -   **Event Map:** Displays the geographic distribution of earthquakes using a scatter map.
    -   **Heat Map:** Shows areas with the highest concentration of seismic activity using a density map.
    -   **Cluster Analysis:** Identifies groups of earthquakes that may be geographically related using the DBSCAN algorithm.
-   **Temporal Analysis:**
    -   **Daily Evolution:** Analyzes the daily evolution of seismic activity, including event counts and magnitude trends.
    -   **Weekly Patterns:** Identifies weekly patterns in seismic activity using bar charts and heatmaps.
    -   **Hourly Patterns:** Examines hourly patterns in seismic activity using bar charts and heatmaps.
-   **Advanced Analysis:**
    -   **Correlations:** Calculates and visualizes correlations between different seismic parameters using a correlation matrix.
    -   **Magnitude by Region:** Analyzes magnitude statistics by region, highlighting regions with significant seismic activity.
    -   **Comparisons:** Compares different seismic variables using scatter plots and statistical summaries.
-   **Data Table:**
    -   Displays the filtered data in a sortable and downloadable table.

## Technologies Used

-   **Streamlit:** A Python library for creating interactive web applications.
-   **Plotly Express:** A high-level Python visualization library for creating interactive charts and maps.
-   **Pandas:** A data analysis and manipulation library.
-   **NumPy:** A library for numerical computing.
-   **Scikit-learn:** A machine learning library used for cluster analysis.

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install streamlit pandas plotly scikit-learn
    ```

3.  **Download the seismic data:**

    -   Obtain the `all_month.csv` file from the [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/earthquake-hazards).
    -   Place the file in the `data/` directory.

4.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

5.  **Access the dashboard in your web browser:**

    -   Streamlit will provide a local URL (e.g., `http://localhost:8501`) to access the dashboard.

## Data Source

-   [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/earthquake-hazards)

## Contributing

Contributions to the Seismic Activity Dashboard are welcome! If you have suggestions for new features, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

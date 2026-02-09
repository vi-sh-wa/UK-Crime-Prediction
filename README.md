
# Distributed Time-Series Forecasting: UK Violent Crime Pipeline
**A Scalable PySpark & SARIMA Implementation for Big Data Analytics**

## Project Overview
This project was developed as part of the **Big Data & Cloud Computing** module at my University. The objective was to analyze the impact of COVID-19 lockdown restrictions on reported violent crimes across the UK using a dataset of over **550,000 records**.

While this originated as an academic project, this repository features a **fully refactored, production-grade implementation**. I have transitioned the original exploratory analysis into a modular Python framework to demonstrate professional data engineering standards.

## Tech 
* **Engine:** Apache Spark (PySpark)
* **Environment:** Local Spark Session (Optimized for multi-core processing)
* **Time-Series Modeling:** Statsmodels (SARIMA)
* **Data Handling:** Pandas, NumPy
* **Visualizations:** Matplotlib, Seaborn

## Data Source & Reproducibility
* The original data was hosted in a private Azure Blob Storage bucket which is no longer active. However, this pipeline is built to be reproducible. Researchers can download the public street-level crime datasets directly from [data.police.uk](https://data.police.uk/) to run the pipeline



## Engineering Enhancements (Standing Out)
To move beyond a standard classroom assignment, I refactored the code to include:
* **Modular Pipeline Architecture:** Replaced linear notebook cells with a Class-based structure (`UKCrimePipeline`) for better scalability and code reuse.
* **Efficient Resource Management:** Configured the Spark Session with `local[*]` and enabled **Arrow optimization** to maximize local CPU utilization during heavy aggregations.
* **Automated Logging:** Integrated the `logging` library to provide real-time telemetry on ETL stages and model training.
* **Hybrid Analysis:** Used Spark SQL for high-volume data pruning and aggregation, then seamlessly transitioned to `statsmodels` for complex SARIMA forecasting.

## Key Analysis Phases
1.  **Distributed ETL:** Cleaning and filtering 550k+ records to isolate violent crime categories.
2.  **Exploratory Data Analysis (EDA):** Visualizing crime trends relative to the March 2020 lockdown.
3.  **Stationarity Testing:** Implementing Augmented Dickey-Fuller (ADF) tests and rolling statistics.
4.  **Forecasting:** Training a SARIMA (Seasonal Autoregressive Integrated Moving Average) model to predict crime volumes.

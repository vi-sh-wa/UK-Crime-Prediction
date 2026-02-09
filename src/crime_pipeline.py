import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isin
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UKCrimePipeline:
    def __init__(self, app_name="UK_Crime_Analysis"):
        """Initialize Spark Session with optimized configurations."""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        logger.info("Spark Session established.")

    def load_and_filter_data(self, data_path):
        """Load Big Data and filter for violent crimes."""
        logger.info(f"Loading data from: {data_path}")
        df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        
        violent_types = ['Violent crime', 'Violence and sexual offences']
        processed_df = df.filter(col("Crime type").isin(violent_types)) \
                         .groupBy("Month").count() \
                         .orderBy("Month")
        
        # Convert to Pandas for Time-Series Analysis (Once its small enough after aggregation)
        v_crime = processed_df.toPandas()
        v_crime['Month'] = pd.to_datetime(v_crime['Month'])
        v_crime.set_index('Month', inplace=True)
        return v_crime

    def perform_trend_analysis(self, df, window=12):
        """Analyze trends using rolling windows."""
        logger.info(f"Performing {window}-month trend analysis.")
        df[f'rolling_mean_{window}'] = df['count'].rolling(window=window).mean()
        # Save plot for further analysis
        return df

    def check_stationarity(self, timeseries):
        """Perform Augmented Dickey-Fuller test."""
        logger.info("Running Augmented Dickey-Fuller Test.")
        result = adfuller(timeseries.dropna())
        logger.info(f'ADF Statistic: {result[0]}')
        logger.info(f'p-value: {result[1]}')
        return result

    def train_forecast_model(self, train_data, order=(1,1,1), s_order=(1,1,1,12)):
        """Train SARIMA model and generate 12-month forecast."""
        logger.info("Fitting SARIMA model...")
        model = SARIMAX(train_data, order=order, seasonal_order=s_order)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=12)
        return results, forecast

    def run_pipeline(self, data_path):
        """Execute the full end-to-end workflow."""
        v_crime = self.load_and_filter_data(data_path)
        
        pre_lockdown = v_crime[v_crime.index < "2020-04-01"] # Lockdown Analysis
        logger.info(f"Avg Pre-Lockdown Crimes: {pre_lockdown['count'].mean():.2f}")

        self.perform_trend_analysis(v_crime) # Stationarity & Trends
        self.check_stationarity(v_crime['count'])

        results, forecast = self.train_forecast_model(pre_lockdown['count']) #Forecasting
        logger.info("Pipeline execution successfully completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Spark Pipeline for UK Crime Analysis")
    parser.add_argument("--data", required=True, help="Path to the crime dataset CSV")
    args = parser.parse_args()

    pipeline = UKCrimePipeline()
    pipeline.run_pipeline(args.data)

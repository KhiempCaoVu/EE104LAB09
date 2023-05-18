# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:29:04 2023

@author: khiem
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the 6-month dataset
data_subset = pd.read_csv('Death_6months.csv')
data_subset['Date'] = pd.to_datetime(data_subset['Date'])
data_subset = data_subset.set_index('Date')

# Load the full-year dataset
data_full = pd.read_csv('Death_12months.csv')
data_full['Date'] = pd.to_datetime(data_full['Date'])
data_full = data_full.set_index('Date')

# Check for missing dates
date_range = pd.date_range(start=data_subset.index.min(), end=data_subset.index.max())
missing_dates = date_range[~date_range.isin(data_subset.index)]
if len(missing_dates) > 0:
    print("Missing dates found: ", missing_dates)

# Fit the SARIMA model on the 6-month data
model_LTCF = sm.tsa.statespace.SARIMAX(data_subset['LTCF'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_LTCF_fit = model_LTCF.fit()
pred_LTCF = model_LTCF_fit.predict(start=data_full.index.min(), end=data_full.index.max())

# Fit the SARIMA model on the 6-month data
model_Non_ltcf = sm.tsa.statespace.SARIMAX(data_subset['Non_ltcf'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_Non_ltcf_fit = model_Non_ltcf.fit()
pred_Non_ltcf = model_Non_ltcf_fit.predict(start=data_full.index.min(), end=data_full.index.max())

# Fit the SARIMA model on the 6-month data 
model_Cumulative = sm.tsa.statespace.SARIMAX(data_subset['Cumulative'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_Cumulative_fit = model_Cumulative.fit()
pred_Cumulative = model_Cumulative_fit.predict(start=data_full.index.min(), end=data_full.index.max())

# Create a new figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the 6-month prediction
ax.plot(data_subset.index, data_subset['LTCF'], label='Actual Data (6 Months)')
ax.plot(data_full.index, pred_LTCF, label='Predicted Total Cases')

ax.plot(data_subset.index, data_subset['Non_ltcf'], label='Actual Data (6 Months)')
ax.plot(data_full.index, pred_Non_ltcf, label='Predicted New Cases')

ax.plot(data_subset.index, data_subset['Cumulative'], label='Actual Data (6 Months)')
ax.plot(data_full.index, pred_Cumulative, label='Predicted 7-day Average New Cases')

# Plot the full-year actual data
ax.plot(data_full.index, data_full['LTCF'], label='Actual Total Cases')
ax.plot(data_full.index, data_full['Non_ltcf'], label='Actual New Cases')
ax.plot(data_full.index, data_full['Cumulative'], label='Actual 7-day Average New Cases')

# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Number of Cases')
ax.set_title('COVID-19 Trend Prediction for Santa Clara County')

# Show the legend
ax.legend()

# Display the plot
plt.grid(True)
plt.show()
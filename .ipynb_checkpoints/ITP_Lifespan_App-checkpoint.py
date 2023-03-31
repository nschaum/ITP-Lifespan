import pandas as pd
import streamlit as st
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from lifelines.statistics import logrank_test

# Read data
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2016_concat.csv'
df = pd.read_csv(data_path)

# Drop rows where the 'group' column is "MetRapa", "Rapa_hi_cycle", or "Rapa_hi_start_stop"
# This will leave only treatments that were applied continuously, and simplify things for the purposes of this app
# We'll make another app later that includes those special cases
df = df.drop(df[df['group'].isin(['MetRapa', 'Rapa_hi_cycle', 'Rapa_hi_start_stop'])].index)

# Convert 'Rx(ppm)' to float with 1 decimal place
df['Rx(ppm)'] = df['Rx(ppm)'].astype(float).round(1)

# Convert 'age_initiation(mo)' to integer
df['age_initiation(mo)'] = df['age_initiation(mo)'].astype(int)

# Define unique combinations of treatments, Rx, cohort, and age_initiation(mo)
unique_treatments = df['treatment'].unique()

rx_age_cohort = {}
for treatment in unique_treatments:
    df_treatment = df[df['treatment'] == treatment]
    unique_rx = df_treatment['Rx(ppm)'].unique()
    unique_cohort = df_treatment['cohort'].unique()
    unique_age_initiation = df_treatment['age_initiation(mo)'].unique()
    rx_age_cohort[treatment] = [(rx, age, cohort) for rx in unique_rx for age in unique_age_initiation for cohort in unique_cohort]

# Define function to filter dataframe based on user input
def filter_dataframe(df, treatment, rx, age_initiation, cohort):
    df_filtered = df[(df['treatment'] == treatment) & (df['Rx(ppm)'] == rx) & (df['age_initiation(mo)'] == age_initiation) & (df['cohort'] == cohort)]
    return df_filtered

# Define global variable
statistics_table = pd.DataFrame()

# Define function to create Kaplan Meier plot and statistics table
def create_km_plot(df, treatment, rx, age_initiation, cohort):
    # Filter dataframe
    df_filtered = filter_dataframe(df, treatment, rx, age_initiation, cohort)

    # Create Kaplan Meier curves
    kmf_treatment = KaplanMeierFitter()
    kmf_treatment.fit(df_filtered['age(days)'], event_observed=(df_filtered['status'] == 'dead'), label=treatment)

    # Plot Kaplan Meier curves
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_treatment.plot(ax=ax, ci_show=False)

    # Set plot title and axis labels
    title = f'{treatment}'
    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Survival Probability')

    # Add vertical line at age_initiation(mo) for treatment curve (except if treatment=control)
    if treatment != 'Control':
        age_initiation_days = age_initiation * 30.4  # convert age_initiation(mo) to days
        ax.axvline(x=age_initiation_days, color='black', linestyle='--',
                   label=None)

    # Add vertical line at y=0.5 for both curves
    ax.axhline(y=0.5, color='gray', linestyle='--', label=None)
    ax.axvline(x=kmf_treatment.median_survival_time_, ymax=0.5, color='gray', linestyle='--',
               label=None)

    # Set legend to the right of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set gridlines
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.grid(which='both', alpha=0.5)

    # Return figure object
    return fig

# Define UI elements
treatment = st.selectbox('Select a treatment', unique_treatments)
treatment_options = rx_age_cohort[treatment]
rx, age_initiation, cohort = st.selectbox('Select the specific cohort (year, dose, age of initiation)', treatment_options)

# Call function to create plot and statistics table, and pass figure object to st.pyplot()
fig = create_km_plot(df, treatment, rx, age_initiation, cohort)
st.pyplot(fig)
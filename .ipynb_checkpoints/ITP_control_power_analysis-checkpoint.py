# August 6, 2023; Nick Schaum

# This notebook takes as input all ITP data from 2004-2017 that I have concatenated into one file.
# It then plots the control data as a Kaplan-Meier curve and calculates some statistics
# The goal is to understand what the average HET3 lifespan is, and the variability in lifespan
# in order to perform power analysis for designing a new lifespan experiment.

### IMPORT REQUIRED PACKAGES AND DATA
import pandas as pd
import streamlit as st
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from lifelines.statistics import logrank_test

# Read concatenated and cleaned raw data file
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2017_concat.csv'
df = pd.read_csv(data_path)

# Read the logrank data
logrank_data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_logrank.csv'
logrank_df = pd.read_csv(logrank_data_path)

# Streamlit app
st.title('Kaplan Meier Curve Generator')

# Initialize an empty DataFrame for plot_data
plot_data = pd.DataFrame()

# Menu items for selecting the number of curves for the user to select, and for adding confidence intervals, or plotting the x-axis in months
n = st.sidebar.number_input("Number of curves to plot", min_value=1, value=2, step=1)
show_confidence_interval = st.sidebar.checkbox('Show confidence interval', value=True)
x_axis_months = st.sidebar.checkbox('Display x-axis in months')

treatment_groups = []
combine_data_flags = []

for i in range(n):
    st.sidebar.header(f'Curve {i + 1}')

    # Treatment selection
    treatments = sorted(df['treatment'].unique())
    default_treatments = ["Rapa", "Control"]
    selected_treatment = st.sidebar.selectbox(f'Select treatment {i + 1}:', treatments, key=f'treatment{i}', index=treatments.index(default_treatments[i]))


    # Filter the data based on selected treatment
    filtered_data = df[df['treatment'] == selected_treatment]

    # Create a DataFrame with unique combinations of cohort, Rx(ppm), and age_initiation(mo)
    unique_combinations = filtered_data[['cohort', 'Rx(ppm)', 'age_initiation(mo)']].drop_duplicates()

    # Display unique combinations
    st.sidebar.write(f'Unique combinations for {selected_treatment}:')

    # Create a custom table with checkboxes
    header_columns = st.sidebar.columns(3)
    header_columns[0].write('Cohort')
    header_columns[1].write('Rx(ppm)')
    header_columns[2].write('Age_initiation(mo)')

    selected_conditions = []
    for index, row in unique_combinations.iterrows():
        cols = st.sidebar.columns([0.2, 1, 1, 1])
        default_checkbox_state = row['cohort'].startswith("C2015") if i < 2 else False
        selected = cols[0].checkbox('', key=f'{index}{i}', value=default_checkbox_state)
        cols[1].write(row['cohort'])
        cols[2].write(row['Rx(ppm)'])
        cols[3].write(row['age_initiation(mo)'])
        if selected:
            selected_conditions.append(index)

    selected_conditions_original = unique_combinations.loc[selected_conditions].to_dict('records')

    # Condition selection
    # Combine data option
    combine_data = st.sidebar.checkbox(f'Combine data for {selected_treatment}', key=f'combine{i}')

    treatment_groups.append((selected_treatment, selected_conditions_original))
    combine_data_flags.append(combine_data)

def filter_data(df, treatment, condition, combine_data):
    if combine_data:
        filtered_data = df[(df['treatment'] == treatment) &
                           (df['cohort'] == condition['cohort']) &
                           (df['Rx(ppm)'] == condition['Rx(ppm)']) &
                           (df['age_initiation(mo)'] == condition['age_initiation(mo)'])]
    else:
        filtered_data = df[(df['treatment'] == treatment) &
                           (df['cohort'] == condition['cohort']) &
                           (df['Rx(ppm)'] == condition['Rx(ppm)']) &
                           (df['age_initiation(mo)'] == condition['age_initiation(mo)'])]
    return filtered_data

# Store selected data in a list
selected_data_list = []

for i, (treatment, conditions) in enumerate(treatment_groups):
    combine_data = combine_data_flags[i]
    combined_data = pd.DataFrame()  # Initialize an empty DataFrame for combined_data
    for condition in conditions:
        selected_data = filter_data(df, treatment, condition, combine_data)
        if combine_data:
            combined_data = combined_data.append(selected_data)  # Append selected_data to combined_data
        else:
            selected_data_list.append(selected_data)

    if combine_data:
        selected_data_list.append(combined_data)  # Add combined_data to selected_data_list

# Reset the index of plot_data DataFrame
plot_data.reset_index(drop=True, inplace=True)

# Kaplan Meier Curve
# Initialize a list to store the labels of each curve
curve_labels = []
if selected_data_list:
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize a list to store the results of each curve
    results = []
    survival_objects = []

    for treatment_data in selected_data_list:
        treatment = treatment_data.iloc[0]['treatment']
        cohort = treatment_data.iloc[0]['cohort']
        rx_ppm = f"{treatment_data.iloc[0]['Rx(ppm)']} ppm"
        age_initiation = f"{treatment_data.iloc[0]['age_initiation(mo)']} mo initiation"
        curve_label = f"{treatment}, {cohort}, {rx_ppm}, {age_initiation}"
        curve_labels.append(curve_label)

        # Fit the Kaplan-Meier estimator and plot the survival curve
        kmf.fit(treatment_data['age(days)'], treatment_data['dead'], label=curve_label)
        kmf.plot(ax=ax, ci_show=show_confidence_interval)

        # Add vertical line for age of initiation if not a Control group
        if treatment != "Control":
            age_initiation_days = treatment_data.iloc[0]['age_initiation(mo)'] * 30
            survival_prob_at_initiation = kmf.predict(age_initiation_days)
            curve_color = ax.get_lines()[-1].get_color()
            ax.plot([age_initiation_days, age_initiation_days], [0, survival_prob_at_initiation], linestyle=':', color=curve_color, alpha=0.8)

        curve_labels.append(curve_label)
        survival_objects.append(kmf.survival_function_)

        # Compute the median and maximum lifespan for the curve
        median_lifespan = kmf.median_survival_time_
        max_lifespan = treatment_data['age(days)'].max()
        results.append((curve_label, median_lifespan, max_lifespan))

        # Add vertical line at the median lifespan
        ax.axvline(median_lifespan, ymin=0, ymax=0.5, linestyle='--', color='gray', alpha=0.8)

    # Convert x-axis to months if the checkbox is selected
    if x_axis_months:
        ax.set_xticklabels(np.round(ax.get_xticks() / 30).astype(int))

    # Move the legend to the right of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1, borderpad=1)

    # Plot styling
    ax.set_title('Kaplan-Meier Survival Curve')
    ax.set_xlabel('Age (months)' if x_axis_months else 'Age (days)')
    ax.set_ylabel('Survival Probability')
    ax.grid()
    
    # Set gridlines
    ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Add a horizontal line at y=0.5 with the same style as the median lifespan vertical lines
    median_line_color = 'gray'
    median_line_width = 1
    median_line_alpha = 0.8
    ax.axhline(y=0.5, color=median_line_color, linewidth=median_line_width, alpha=median_line_alpha, linestyle='--')

    # Calculate pairwise log-rank test results
    logrank_results = []
    for i in range(len(survival_objects)):
        for j in range(i + 1, len(survival_objects)):
            result = logrank_test(survival_objects[i], survival_objects[j])
            p_value = result.p_value
            logrank_results.append((i + 1, j + 1, p_value))

    # Create a DataFrame for log-rank test results
    logrank_df = pd.DataFrame([(curve_labels[i], curve_labels[j], p_value) for i, j, p_value in logrank_results], columns=["Curve 1", "Curve 2", "P-value"])


    # Display the plot in the Streamlit app
    st.pyplot(fig)

    # Create a table with the median and maximum lifespan for each curve
    st.write('## Results')
    results_df = pd.DataFrame(results, columns=['Label', 'Median Lifespan', 'Max Lifespan'])
    results_df.index += 1  # Increase index value by 1
    results_df.index.name = 'Curve'
    st.write(results_df)
    
    st.write("## Log-Rank Test Results")
    st.write(logrank_df)

else:
    st.warning('No data available for the selected options.')

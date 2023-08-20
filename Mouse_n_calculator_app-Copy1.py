##This is a more sophisticated calculation of required number of starting mice that uses bootstrapping. 

import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from math import ceil

# Load data from the specified path on your machine
data = pd.read_csv("C:/Users/ndsch/Data/ITP-Lifespan-Data/ITP_processed_data/ITP_2004-2017_concat.csv")

# Initial filter for Control treatment
control_data = data[data['treatment'] == 'Control']

def filter_data_by_sex_and_cohort(data, selected_sex, selected_cohorts):
    # Adjust for the representation in the dataset
    sex_mapping = {
        "Male": "m",
        "Female": "f"
    }
    
    # Filter by selected sex
    data = data[data['sex'] == sex_mapping[selected_sex]]
    
    # Filter by selected cohorts
    if "All" not in selected_cohorts:
        data = data[data['cohort'].isin(selected_cohorts)]
    
    return data

def calculate_required_mice(n_mice, purchase_age, desired_age, filtered_data):
    kmf = KaplanMeierFitter()
    kmf.fit(filtered_data['age(days)'], event_observed=filtered_data['dead'], timeline=np.arange(0, max(filtered_data['age(days)'])+1, 1))

    # Survival probability at purchase age
    surv_prob_at_purchase = kmf.predict(purchase_age)
    # Survival probability at desired age
    surv_prob_at_desired_age = kmf.predict(desired_age)

    # Adjusted survival probability at desired age
    adjusted_surv_prob = surv_prob_at_desired_age / surv_prob_at_purchase

    # Calculate required mice
    required_mice = ceil(n_mice / adjusted_surv_prob)

    return required_mice, adjusted_surv_prob

def plot_survival_curve(ax, data, sex):
    kmf = KaplanMeierFitter()
    kmf.fit(data['age(days)'], event_observed=data['dead'])
    kmf.plot(ax=ax, label=f'{sex} (n={len(data)}) Median: {int(kmf.median_survival_time_)} days')

def bootstrap_estimate(n_mice, required_mice, purchase_age, desired_age, data, n_iterations=10):
    required_mice_samples = []
    median_survivals = []  # To store median survivals
    datasets = []  # To store bootstrapped datasets

    for _ in range(n_iterations):
        sample_data = data[data['age(days)'] >= purchase_age].sample(n=int(required_mice), replace=True)
        
        # Calculate the number of mice that reached the desired age in this sample
        mice_reaching_desired_age = sum(sample_data['age(days)'] >= desired_age)
        required_mice_sample = n_mice / (mice_reaching_desired_age / required_mice)
        required_mice_samples.append(required_mice_sample)

        kmf = KaplanMeierFitter()
        kmf.fit(sample_data['age(days)'], event_observed=sample_data['dead'])
        median_survivals.append(kmf.median_survival_time_)
        datasets.append(sample_data)
    
    # Identify datasets with best and worst median survival
    best_dataset = datasets[np.argmax(median_survivals)]
    worst_dataset = datasets[np.argmin(median_survivals)]

    return pd.Series(required_mice_samples), best_dataset, worst_dataset

def main():
    st.title("Mouse Sample Size Forcaster")

    # Add the introduction paragraph
    st.markdown("""Given a specific number of mice needed at a desired age, this app calculates the number of mice you should initially start with. This prediction is based on survival data from control mice (HET3s) used by the Interventions Testing Program (ITP) from cohorts 2004-2017. You may optionally select an individual cohort or any combination of cohorts instead of all cohorts. Note that the 2017 has unusually low survival, so it may be unwise (or perhaps wise if one wants to be conservative) to use that cohort.""")
    
    # Multi-select dropdown for selecting cohort(s)
    available_cohorts = ["All"] + sorted(control_data['cohort'].unique().tolist())
    selected_cohorts = st.multiselect("Select Cohort(s):", available_cohorts, default=["All"])

    # User Inputs using input boxes
    purchase_age = st.number_input("Enter Starting Age of Mice (in days):", value=100)
    desired_age = st.number_input("Enter Desired Age of Survival for Mice (in days):", value=913)
    n_mice = st.number_input("Enter Number of Mice You Want at Desired Age:", value=8)

    if "calculate_pressed" not in st.session_state:
        st.session_state.calculate_pressed = False
    
    if st.button("Calculate") or st.session_state.calculate_pressed:
        # Initial estimate calculations
        results = []
        fig, ax = plt.subplots(figsize=(10, 7))
        required_mice_dict = {}  # A dictionary to store required mice for each sex
        for sex_option in ["Male", "Female"]:
            filtered_data = filter_data_by_sex_and_cohort(control_data, sex_option, selected_cohorts)
            required_mice, _ = calculate_required_mice(n_mice, purchase_age, desired_age, filtered_data)
            required_mice_dict[sex_option] = np.ceil(required_mice)  # Rounding up and storing for each sex
            results.append([sex_option, ceil(required_mice)])
            plot_survival_curve(ax, filtered_data, sex_option)
        
        ax.set_title('Survival Data Used to Calculate Number of Required Mice')
        ax.set_ylabel('Survival Probability')
        ax.set_xlabel('Days')
        ax.legend()
        st.pyplot(fig)
        
        result_df = pd.DataFrame(results, columns=["Sex", "Required Mice"])
        result_df.set_index("Sex", inplace=True)
        st.table(result_df)

        st.session_state.calculate_pressed = True
        

    if "calculate_pressed" in st.session_state:
        bootstrap_cycles = st.number_input("Number of Bootstrap Cycles:", value=10)

        if st.button("Bootstrap"):
            # Bootstrapping
            bootstrap_results = []

            st.markdown("""
            Bootstrapping is a technique used to simulate the variability one might expect in an actual experiment by drawing repeated random samples from a dataset. In this case, we are using the survival data, as visualized in the Kaplan-Meier curve above, to understand potential variation in our experiment outcomes. 

            For our bootstrap analysis, we take a random subsample from the selected data, with a size equivalent to our initial estimate of required mice (as calculated above). This acts as a simulated experiment, where we start with the estimated number of mice and observe how many reach the desired age. By doing this thousands of times, we simulate the experiment under many different scenarios to understand the potential range of outcomes.

            The table below shows the results of this bootstrapping. The "2.5th Percentile" and "97.5th Percentile" provide what is often termed a 95% confidence interval. This interval gives a range in which we can be 95% confident that the actual number of required mice will fall, were we to conduct the experiment many times.
            """)

            # Setting up a figure for histograms
            fig_hist, axes_hist = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))  # Setting up a figure for Kaplan-Meier curves
            
            for idx, sex_option in enumerate(["Male", "Female"]):
                filtered_data = filter_data_by_sex_and_cohort(control_data, sex_option, selected_cohorts)
                bootstrap_series, best_data, worst_data = bootstrap_estimate(n_mice, required_mice_dict[sex_option], purchase_age, desired_age, filtered_data, n_iterations=bootstrap_cycles)
                bootstrap_results.append([
                    sex_option, 
                    ceil(bootstrap_series.median()), 
                    ceil(bootstrap_series.quantile(0.025)), 
                    ceil(bootstrap_series.quantile(0.80)),
                    ceil(bootstrap_series.quantile(0.90)),
                    ceil(bootstrap_series.quantile(0.95)),
                    ceil(bootstrap_series.quantile(0.975))
                ])

                
                # Plotting histogram using the stored results
                bootstrap_min = int(bootstrap_series.min())
                bootstrap_max = int(bootstrap_series.max())
                axes_hist[idx].hist(bootstrap_series, bins=range(bootstrap_min, bootstrap_max + 1), color='skyblue', edgecolor='black')
                axes_hist[idx].axvline(bootstrap_series.median(), color='red', linestyle='dashed', linewidth=1)
                axes_hist[idx].set_title(f'{sex_option} Required Mice Distribution')
                axes_hist[idx].set_xlabel('Required Mice')
                axes_hist[idx].set_ylabel('Frequency')
                min_ylim, max_ylim = plt.ylim()

                # Plotting the Kaplan-Meier curves for best and worst median survival for each sex
                kmf = KaplanMeierFitter()
                kmf.fit(best_data['age(days)'], event_observed=best_data['dead'])
                kmf.plot(ax=axes[idx], label=f'Best Median Survival (n={len(best_data)}): {int(kmf.median_survival_time_)} days')

                kmf = KaplanMeierFitter()
                kmf.fit(worst_data['age(days)'], event_observed=worst_data['dead'])
                kmf.plot(ax=axes[idx], label=f'Worst Median Survival (n={len(worst_data)}): {int(kmf.median_survival_time_)} days')

                axes[idx].set_title(f'{sex_option} Kaplan-Meier Curves for Best and Worst Median Survival from Bootstrapping')
                axes[idx].set_xlim([purchase_age, desired_age])  # This line adjusts the x-axis range
                axes[idx].set_ylabel('Survival Probability')
                axes[idx].set_xlabel('Days')
                axes[idx].legend()

            # Create and show the table
            bootstrap_df = pd.DataFrame(bootstrap_results, columns=["Sex", "Median Required Mice", "2.5th Percentile", "80th", "90th", "95th", "97.5th"])
            bootstrap_df.set_index("Sex", inplace=True)
            st.table(bootstrap_df)

            # Show the histograms
            st.pyplot(fig_hist)

            # Show the Kaplan-Meier curves
            st.pyplot(fig)
            
if __name__ == "__main__":
    main()

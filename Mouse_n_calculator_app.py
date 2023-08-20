### This is a fully functional app. Don't change the code! It works to estimate the number of starting mice based on desire mice of a given age.
### 2023-08-09; Nick Schaum

import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

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
    required_mice = n_mice / adjusted_surv_prob

    return required_mice, adjusted_surv_prob

def plot_survival_curve(ax, data, sex):
    kmf = KaplanMeierFitter()
    kmf.fit(data['age(days)'], event_observed=data['dead'])
    kmf.plot(ax=ax, label=f'{sex} (n={len(data)}) Median: {int(kmf.median_survival_time_)} days')

def main():
    st.sidebar.title("Mouse Sample Size Forcaster")

    # Add the introduction paragraph
    st.markdown("""
    This program computes an adjusted survival probability derived by comparing the likelihood of survival at two specific age points: the age at which the mice are procured (purchase age) and the age at which they're desired to still be alive (desired age): 
    
    adjusted_surv_prob = surv_prob_at_desired_age / surv_prob_at_purchase

    This adjusted probability represents the conditional chance that a mouse, once reaching the purchase age, will continue to survive up to the desired age. The program then calculates the initial number of mice required to ensure that a certain number (n_mice) survive until the desired age, using the formula: 
    
    required_mice = n_mice / adjusted_surv_prob

    **Warning:** This is for ballpark estimations only. Mouse lifespan is highly variable and this estimate will not guarantee with any confidence that the estimated number of starting mice will be sufficient. It is based on survival probability from thousdands of mice, and smaller cohorts are subject to wide variability in median lifespan.
    """)


    st.sidebar.markdown("Given a specific number of mice needed at a desired age, this app calculates the number of mice you should initially start with. This prediction is based on survival data from control mice (HET3s) used by the Interventions Testing Program (ITP) from cohorts 2004-2017. You may optionally select an individual cohort or any combination of cohorts instead of all cohorts. Note that the 2017 has unusually low survival, so it may be unwise (or perhaps wise if one wants to be conservative) to use that cohort.")

    # Multi-select dropdown for selecting cohort(s) in the sidebar
    available_cohorts = ["All"] + sorted(control_data['cohort'].unique().tolist())
    selected_cohorts = st.sidebar.multiselect("Select Cohort(s):", available_cohorts, default=["All"])

    # User Inputs using input boxes in the sidebar
    purchase_age = st.sidebar.number_input("Enter Starting Age of Mice (in days):", value=100)
    desired_age = st.sidebar.number_input("Enter Desired Age of Mice (in days):", value=913)
    n_mice = st.sidebar.number_input("Enter Number of Mice Per Sex Desired at That Age:", value=8)

    if st.sidebar.button("Calculate"):
        results = []

        fig, ax = plt.subplots(figsize=(10, 7))
        for sex_option in ["Male", "Female"]:
            filtered_data = filter_data_by_sex_and_cohort(control_data, sex_option, selected_cohorts)
            required_mice, _ = calculate_required_mice(n_mice, purchase_age, desired_age, filtered_data)
            results.append([sex_option, required_mice])

            plot_survival_curve(ax, filtered_data, sex_option)

        ax.set_title('Survival Data Used to Calculate Number of Required Mice')
        ax.set_ylabel('Survival Probability')
        ax.set_xlabel('Days')
        ax.legend()
        st.pyplot(fig)

        # Display results in a table format
        result_df = pd.DataFrame(results, columns=["Sex", "Required Mice"])
        result_df.set_index("Sex", inplace=True)  # Set "Sex" as the index to remove default index
        st.table(result_df)

if __name__ == "__main__":
    main()

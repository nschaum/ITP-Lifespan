import streamlit as st
import pandas as pd
import statsmodels.stats.power as smp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import PowerNorm
from lifelines import KaplanMeierFitter
from math import ceil

# Load the lifespan data
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2017_concat.csv'
data = pd.read_csv(data_path)

# Load the mouse purchase data
purchase_data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\Mouse_costs\\JAX_HET3_prices.csv'
purchase_data = pd.read_csv(purchase_data_path)

# Title for the Streamlit app
st.title('The Mouse Aging Tool Suite')

def power_analysis_page():
    st.sidebar.header('Power Analysis')
    alpha = st.sidebar.slider("Select Alpha", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    st.sidebar.write("Here, the two-sample independent t-test with the function smp.TTestIndPower().solve_power from the statsmodels.stats.power module has been employed.")
    power_range = (0.7, 0.90)
    effect_size_range = (0.1, 0.3)
    male_power_values, male_effect_size_values, male_sample_sizes = compute_required_sample_size('m', alpha, power_range, effect_size_range)
    female_power_values, female_effect_size_values, female_sample_sizes = compute_required_sample_size('f', alpha, power_range, effect_size_range)
    vmin_value = min(male_sample_sizes.min(), female_sample_sizes.min())
    vmax_value = max(male_sample_sizes.max(), female_sample_sizes.max())
    norm = PowerNorm(gamma=0.5, vmin=vmin_value, vmax=vmax_value)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))
    sns.heatmap(male_sample_sizes, cmap='YlGnBu', annot=True, fmt=".0f", 
                cbar_kws={'label': 'Required Sample Size'}, 
                ax=ax1, 
                yticklabels=[f"{x:.1%}" for x in male_effect_size_values], 
                xticklabels=[f"{x:.1%}" for x in male_power_values],
                vmin=vmin_value, vmax=vmax_value,
                norm=norm)
    ax1.set_title("Males")
    ax1.set_xlabel("Power (%)")
    ax1.set_ylabel("Desired Effect Size (%)")
    sns.heatmap(female_sample_sizes, cmap='YlGnBu', annot=True, fmt=".0f", 
                cbar_kws={'label': 'Required Sample Size'}, 
                ax=ax2, 
                yticklabels=[f"{x:.1%}" for x in female_effect_size_values], 
                xticklabels=[f"{x:.1%}" for x in female_power_values],
                vmin=vmin_value, vmax=vmax_value,
                norm=norm)
    ax2.set_title("Females")
    ax2.set_xlabel("Power (%)")
    ax2.set_ylabel("Desired Effect Size (%)")
    plt.tight_layout()
    st.pyplot(fig)

# Cost Estimation Page Code
def cost_estimation_page():
    st.sidebar.subheader('Mouse Purchase Cost Calculator')
    mice_per_dose = st.sidebar.number_input('Mice per dose', min_value=1, value=200)
    doses_per_treatment = st.sidebar.number_input('Number of doses per treatment', min_value=1, value=1)
    number_of_treatments = st.sidebar.number_input('Number of treatments', min_value=1, value=1)
    
    pricing_method = st.sidebar.radio("Choose pricing method:", ["Age-based pricing", "Custom price"])
    
    if pricing_method == "Age-based pricing":
        # Age selection with multiple units
        age_unit = st.sidebar.selectbox('Choose age unit for purchase:', ['days', 'weeks', 'months'])
        if age_unit == 'days':
            age_value = st.sidebar.number_input('Age in days at purchase', min_value=1, max_value=78*7)
        elif age_unit == 'weeks':
            age_value = st.sidebar.number_input('Age in weeks at purchase', min_value=1, max_value=78)
            age_value *= 7  # Convert weeks to days
        else:
            age_value = st.sidebar.number_input('Age in months at purchase', min_value=1, max_value=int(78/4))
            age_value *= 30.41666666  # Convert months to days (approximation)

        # Find the corresponding week for pricing. 
        # We use the ceil function to ensure that if the age exceeds a week boundary, 
        # we consider the price of the next week.
        corresponding_week = -(-age_value // 7)

        # Extracting the price for the corresponding week
        if corresponding_week > 78:  # Ensure we don't go beyond the maximum available week
            corresponding_week = 78
        prices_for_week = purchase_data[purchase_data['Age (weeks)'] == corresponding_week]['Price'].values
        if len(prices_for_week) > 0:
            mouse_price = prices_for_week[0]
        else:
            st.sidebar.warning(f"No price available for {corresponding_week} weeks. Please choose a different age.")
            return

    else:
        mouse_price = st.sidebar.number_input('Enter custom price per mouse ($)', min_value=0.0, value=20.0)

    total_mice_needed = mice_per_dose * doses_per_treatment * number_of_treatments
    total_purchase_cost = total_mice_needed * mouse_price

    st.sidebar.write(f"Total mice needed: {total_mice_needed}")
    st.sidebar.write(f"Cost per mouse: ${mouse_price:.2f}")
    st.sidebar.write(f"Total cost to purchase all mice: ${total_purchase_cost:,.2f}")
    
    st.sidebar.subheader('Mouse Housing Cost Calculator')
    mice_per_cage = st.sidebar.number_input('Enter number of mice per cage', min_value=1, value=5)
    
    # Input for days to house mice with option to choose unit
    time_unit = st.sidebar.selectbox('Choose time unit:', ['days', 'weeks', 'months'])
    if time_unit == 'days':
        time_value = st.sidebar.number_input('Enter number of days to house mice', min_value=1, value=30)
    elif time_unit == 'weeks':
        time_value = st.sidebar.number_input('Enter number of weeks to house mice', min_value=1, value=4)
        time_value *= 7  # Convert weeks to days
    else:
        time_value = st.sidebar.number_input('Enter number of months to house mice', min_value=1, value=1)
        time_value *= 30.41666666  # Convert months to days (approximation)

    rounded_time_value = round(time_value)
    st.sidebar.write(f"This corresponds to {rounded_time_value} days.")

    daily_cage_cost = st.sidebar.number_input('Enter daily cage cost ($)', min_value=0.0, value=1.50)
    total_cages = -(-total_mice_needed // mice_per_cage)
    housing_cost = total_cages * daily_cage_cost * time_value
    
    st.sidebar.write(f"Total cost to house mice: ${housing_cost:,.2f}")
    
    grand_total = total_purchase_cost + housing_cost
    st.sidebar.subheader(f'Grand Total: ${grand_total:,.2f}')


    # Plotting total cost vs cost per mouse
    mouse_price_range = np.linspace(0, 1000, 200)
    total_costs_purchase = mouse_price_range * total_mice_needed

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(mouse_price_range, total_costs_purchase, color='b')
    ax1.set_title("Total Cost vs. Cost Per Mouse")
    ax1.set_xlabel("Cost Per Mouse ($)")
    ax1.set_ylabel("Total Cost ($)")
    ax1.grid(True)
    st.pyplot(fig)

    # Plotting total housing cost vs daily cage cost
    daily_cage_cost_range = np.linspace(0, 14, 200)  # Assuming a reasonable range for daily cage costs
    total_costs_housing = total_cages * daily_cage_cost_range * time_value

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(daily_cage_cost_range, total_costs_housing, color='r')
    ax2.set_title("Total Housing Cost vs. Daily Cage Cost")
    ax2.set_xlabel("Daily Cage Cost ($)")
    ax2.set_ylabel("Total Housing Cost ($)")
    ax2.grid(True)
    st.pyplot(fig)

    
    
    # Plotting the heatmap of total cost based on daily cage cost and mouse cost
    daily_cage_cost_range = np.arange(2, 21, 2)  # $2, $4, ..., $20
    mouse_price_range = np.arange(50, 501, 50)   # $50, $100, ..., $500
    
    # Initialize an empty matrix to store total costs
    total_costs = np.zeros((len(mouse_price_range), len(daily_cage_cost_range)))
    
    # Populate the matrix with total cost values
    for i, mouse_cost in enumerate(mouse_price_range):
        for j, cage_cost in enumerate(daily_cage_cost_range):
            purchase_cost = mouse_cost * total_mice_needed
            housing_cost = total_cages * cage_cost * time_value
            total_costs[i, j] = (purchase_cost + housing_cost) / 1_000_000  # Convert to millions

    fig, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(total_costs[::-1], cmap='YlGnBu', annot=True, fmt='.1f', 
                xticklabels=daily_cage_cost_range, yticklabels=mouse_price_range[::-1], cbar_kws={'label': 'Total Cost (in Millions)'})
    ax3.set_title("Heatmap of Total Cost")
    ax3.set_xlabel("Daily Cage Cost ($)")
    ax3.set_ylabel("Cost Per Mouse ($)")
    plt.tight_layout()
    st.pyplot(fig)

def compute_required_sample_size(sex_filter, alpha, power_range, effect_size_range):
    power_values = np.linspace(power_range[0], power_range[1], 30)
    effect_size_values = np.linspace(effect_size_range[0], effect_size_range[1], 30)
    
    lifespans = data[(data['sex'] == sex_filter) & (data['group'] == 'Control')]['age(days)']
    mean_lifespan = lifespans.mean()
    std_lifespan = lifespans.std()


    sample_sizes = np.zeros((len(effect_size_values), len(power_values)))
    
    for i, desired_effect_percentage in enumerate(effect_size_values):
        effect_size_days = desired_effect_percentage * mean_lifespan
        cohens_d = effect_size_days / std_lifespan
        
        for j, power in enumerate(power_values):
            sample_size = smp.TTestIndPower().solve_power(effect_size=cohens_d, alpha=alpha, power=power)
            sample_sizes[i, j] = ceil(sample_size)  # Round up to the next integer

    return power_values, effect_size_values, sample_sizes

def bootstrap_page():

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
        st.title("Mouse Sample Size Forecaster")

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
                        ceil(bootstrap_series.quantile(0.80)),
                        ceil(bootstrap_series.quantile(0.90)),
                        ceil(bootstrap_series.quantile(0.95)),
                        ceil(bootstrap_series.quantile(0.99))
                    ])


                    # Plotting histogram using the stored results
                    bootstrap_min = int(bootstrap_series.min())
                    bootstrap_max = int(bootstrap_series.max())
                    axes_hist[idx].hist(bootstrap_series, bins=20, color='skyblue', edgecolor='black')
                    # alternatively use a step size of one in the above line, but note that b/c of the discrete nature of survival probabilities, the histograms will have gaps at certain discrete values:
                    # bins=range(bootstrap_min, bootstrap_max + 1)
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
                bootstrap_df = pd.DataFrame(bootstrap_results, columns=["Sex", "Median", "80th", "90th", "95th", "99th"])
                bootstrap_df.set_index("Sex", inplace=True)
                st.table(bootstrap_df)

                # Show the histograms
                st.pyplot(fig_hist)

                # Show the Kaplan-Meier curves
                st.pyplot(fig)
                        
    if __name__ == "__main__":
        main()
            
# Page selection
page = st.sidebar.radio("Choose a page:", ["Power Analysis & Heatmaps", "Mouse Cost Calculator", "Bootstrap Survival Estimation"])

if page == "Power Analysis & Heatmaps":
    power_analysis_page()
elif page == "Mouse Cost Calculator":
    cost_estimation_page()
elif page == "Bootstrap Survival Estimation":
    bootstrap_page()
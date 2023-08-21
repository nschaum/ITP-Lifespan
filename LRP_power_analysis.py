###2023-08-20
###This is the working app:
###Power analysis and cost estimates

import streamlit as st
import pandas as pd
import statsmodels.stats.power as smp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import PowerNorm
from lifelines import KaplanMeierFitter

# Load the lifespan data
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2017_concat.csv'
data = pd.read_csv(data_path)

# Load the mouse purchase data
purchase_data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\Mouse_costs\\JAX_HET3_prices.csv'
purchase_data = pd.read_csv(purchase_data_path)

# Title for the Streamlit app
st.title('Power Analysis & Cost Estimation App')

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
            sample_sizes[i, j] = smp.TTestIndPower().solve_power(effect_size=cohens_d, 
                                                                 alpha=alpha, 
                                                                 power=power, 
                                                                 ratio=1.0, 
                                                                 alternative='two-sided')
    return power_values, effect_size_values, sample_sizes

# Page selection
page = st.sidebar.radio("Choose a page:", ["Power Analysis & Heatmaps", "Mouse Cost Calculator"])

if page == "Power Analysis & Heatmaps":
    power_analysis_page()
elif page == "Mouse Cost Calculator":
    cost_estimation_page()

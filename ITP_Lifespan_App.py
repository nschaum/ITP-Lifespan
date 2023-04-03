import pandas as pd
import streamlit as st
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from lifelines.statistics import logrank_test

# Read concatenated raw data file
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2016_concat_simple.csv'
df = pd.read_csv(data_path)

# Read the logrank data
logrank_data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_logrank_simple.csv'
logrank_df = pd.read_csv(logrank_data_path)

# Add a new column that combines treatment and full_name values
df["treatment_fullname"] = df["treatment"] + ": " + df["full_name"]

# Streamlit app
st.sidebar.title("ITP Lifespan Browser")
st.sidebar.write("This app allows you to view the survival curve for any* treatment tested by the Interventions Testing Program between 2004-2016.")
st.sidebar.write("*Currently excludes special cases like combined treatments or treatments that were not continuous throughout the remaining lifespan. Future version will include these cases.")

# Get unique treatment_fullname values
treatment_fullname_values = sorted(df["treatment_fullname"].unique())
selected_treatment_fullname = st.sidebar.selectbox("Select a treatment", treatment_fullname_values)

# Get the corresponding treatment value from the selected treatment_fullname
selected_treatment = selected_treatment_fullname.split(": ")[0]

# Get unique combinations of Rx(ppm), age_initiation(mo), and cohort for the selected treatment
unique_combinations = df.loc[df["treatment"] == selected_treatment, ["Rx(ppm)", "age_initiation(mo)", "cohort"]].drop_duplicates()
unique_combinations["combo"] = unique_combinations["Rx(ppm)"].astype(str) + " ppm, " + unique_combinations["age_initiation(mo)"].astype(str) + " mo, " + unique_combinations["cohort"]

# Let the user select a unique combination
selected_combo = st.sidebar.selectbox("Select a cohort (dose, age of initiation, and year)", unique_combinations["combo"].tolist())
selected_rx_ppm, selected_age_initiation, selected_cohort = selected_combo.split(", ")

# Extract the numeric values from the selected_combo
selected_rx_ppm = float(selected_rx_ppm.split(" ")[0])
selected_age_initiation = int(selected_age_initiation.split(" ")[0])

# Let the user select the sex and site
sex_values = ["m", "f", "m+f"]
selected_sex = st.sidebar.selectbox("Select sex (m, f, or m+f)", sex_values, index=2)

site_values = ["TJL", "UM", "UT", "TJL+UM", "TJL+UT", "UM+UT", "TJL+UM+UT"]
selected_site = st.sidebar.selectbox("Select site (TJL, UM, UT, TJL+UM, TJL+UT, UM+UT, or TJL+UM+UT)", site_values, index=6)


# Filter the logrank data based on the user's selections
selected_logrank_row = logrank_df[(logrank_df['treatment'] == selected_treatment) &
                                  (logrank_df['Rx(ppm)'] == selected_rx_ppm) &
                                  (logrank_df['age_initiation(mo)'] == selected_age_initiation) &
                                  (logrank_df['cohort'] == selected_cohort) &
                                  (logrank_df['sex'] == selected_sex) &
                                  (logrank_df['site'] == selected_site)]

logrank_filtered = selected_logrank_row.copy()

# Filter the data based on the user's selections
selected_data = df[(df["treatment"] == selected_treatment) & (df["Rx(ppm)"] == selected_rx_ppm) & (df["age_initiation(mo)"] == selected_age_initiation) & (df["cohort"] == selected_cohort)]

# Apply the sex filter
if selected_sex != "m+f":
    selected_data = selected_data[selected_data["sex"] == selected_sex]

# Apply the site filter
if selected_site != "TJL+UM+UT":
    selected_sites = selected_site.split("+")
    selected_data = selected_data[selected_data["site"].isin(selected_sites)]

# Filter the control data
control_data = df[(df["treatment"] == "Control") & (df["cohort"] == selected_cohort)]

# Apply the sex filter to the control data
if selected_sex != "m+f":
    control_data = control_data[control_data["sex"] == selected_sex]

# Apply the site filter to the control data
if selected_site != "TJL+UM+UT":
    selected_sites = selected_site.split("+")
    control_data = control_data[control_data["site"].isin(selected_sites)]
    
# Kaplan Meier analysis
kmf = KaplanMeierFitter()
kmf.fit(selected_data["age(days)"], event_observed=selected_data["dead"], label=selected_treatment.capitalize() if selected_treatment == 'control' else selected_treatment)
kmf_control = KaplanMeierFitter()
kmf_control.fit(control_data["age(days)"], event_observed=control_data["dead"], label="control")

# Plot the Kaplan Meier curves
fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot(ax=ax)
kmf_control.plot(ax=ax)

ax.set_title(selected_treatment_fullname, fontsize=20)

ax.axhline(y=0.5, linestyle="--", color="gray")
ax.axvline(x=selected_age_initiation * 30, linestyle="--", color="gray")
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.grid(True)

# Add vertical lines at the median survival time
median_lifespan_selected = kmf.median_survival_time_
median_lifespan_control = kmf_control.median_survival_time_

selected_color = ax.get_lines()[0].get_c()
control_color = ax.get_lines()[1].get_c()

ax.axvline(x=median_lifespan_selected, ymin=0, ymax=0.5, linestyle="--", color=selected_color)
ax.axvline(x=median_lifespan_control, ymin=0, ymax=0.5, linestyle="--", color=control_color)

# Add labels
init_label = f"{selected_treatment} start at {selected_age_initiation}mos of age"
ax.text(selected_age_initiation * 30 - 10, 0.01, init_label, rotation=90, va="bottom", ha="right", fontsize=12, color="gray")

# Adds labels on a particular side of the line depending on where the lines are located so the labels don't overlap
if median_lifespan_selected > median_lifespan_control:
    ax.text(median_lifespan_selected + 10, 0.01, f"{median_lifespan_selected:.1f} days", rotation=90, va="bottom", ha="left", fontsize=12, color=selected_color)
    ax.text(median_lifespan_control - 10, 0.01, f"{median_lifespan_control:.1f} days", rotation=90, va="bottom", ha="right", fontsize=12, color=control_color)
else:
    ax.text(median_lifespan_selected - 10, 0.01, f"{median_lifespan_selected:.1f} days", rotation=90, va="bottom", ha="right", fontsize=12, color=selected_color)
    ax.text(median_lifespan_control + 10, 0.01, f"{median_lifespan_control:.1f} days", rotation=90, va="bottom", ha="left", fontsize=12, color=control_color)

ax.legend(loc="upper right", fontsize=14)
st.pyplot(fig)

# Output tables
median_lifespan = pd.DataFrame({"Treatment": [selected_treatment, "control"],
                                 "Median Lifespan (days)": [kmf.median_survival_time_, kmf_control.median_survival_time_],
                                 "Max Lifespan (days)": [selected_data["age(days)"].max(), control_data["age(days)"].max()]})
st.write(median_lifespan.set_index("Treatment"))

# Calculate percentage difference between treatment and control median lifespans
median_lifespan_diff = (kmf.median_survival_time_ - kmf_control.median_survival_time_) / kmf_control.median_survival_time_ * 100
st.markdown(f"**Difference in median lifespan:** {median_lifespan_diff:.0f}%")

# Get Log-rank test results from filtered logrank_df
test_statistic = logrank_filtered['test_statistic'].values[0]
p_value = logrank_filtered['p-value'].values[0]

st.markdown("<u>Log-Rank Test</u>", unsafe_allow_html=True)
st.markdown(f"  Test statistic: {test_statistic:.2f}")
if p_value < 0.00001:
    st.markdown(f"  p-value: {p_value:.1e}")
else:
    st.markdown(f"  p-value: {p_value:.5f}")

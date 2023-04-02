import pandas as pd
import numpy as np
from itertools import product
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Read data
data_path = 'C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_2004-2016_concat_simple.csv'
df = pd.read_csv(data_path)

# Create a function to generate the table
def generate_table(df):
    results = []

    # Iterate over unique combinations of treatment, Rx(ppm), age_initiation(mo), cohort, and sex
    for treatment, rx_ppm, age_initiation, cohort, sex in product(df["treatment"].unique(), df["Rx(ppm)"].unique(), df["age_initiation(mo)"].unique(), df["cohort"].unique(), ["m", "f", "m+f"]):
        treatment_data = df[(df["treatment"] == treatment) & (df["Rx(ppm)"] == rx_ppm) & (df["age_initiation(mo)"] == age_initiation) & (df["cohort"] == cohort)]
        control_data = df[(df["treatment"] == "Control") & (df["cohort"] == cohort)]

        if sex == "m":
            treatment_data = treatment_data[treatment_data["sex"] == "m"]
            control_data = control_data[control_data["sex"] == "m"]
        elif sex == "f":
            treatment_data = treatment_data[treatment_data["sex"] == "f"]
            control_data = control_data[control_data["sex"] == "f"]

        if treatment_data.empty or control_data.empty:
            continue

        kmf_treatment = KaplanMeierFitter()
        kmf_treatment.fit(treatment_data["age(days)"], event_observed=treatment_data["dead"])

        kmf_control = KaplanMeierFitter()
        kmf_control.fit(control_data["age(days)"], event_observed=control_data["dead"])

        logrank_result = logrank_test(treatment_data["age(days)"], control_data["age(days)"], event_observed_A=treatment_data["dead"], event_observed_B=control_data["dead"])

        median_diff = (kmf_treatment.median_survival_time_ - kmf_control.median_survival_time_)

        results.append({
            "Treatment": treatment,
            "Dose": rx_ppm,
            "Age of initiation": age_initiation,
            "Cohort": cohort,
            "Sex": sex,
            "Test statistic": logrank_result.test_statistic,
            "P-value": logrank_result.p_value,
            "Difference in median lifespan": median_diff
        })

    results_df = pd.DataFrame(results)
    return results_df

table = generate_table(df)
output_csv_path = "C:\\Users\\ndsch\\Data\\ITP-Lifespan-Data\\ITP_processed_data\\ITP_logrank_results.csv"
table.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")

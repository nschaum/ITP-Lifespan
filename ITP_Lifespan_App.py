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

st.set_page_config(layout="wide")

# Main menu for switching between pages
st.sidebar.title("ITP Lifespan Browser")
st.sidebar.write("This app allows you to view the survival curve for any* treatment tested by the Interventions Testing Program between 2004-2016.")
st.sidebar.write("*Currently excludes special cases like combined treatments or treatments that were not continuous throughout the remaining lifespan. Future version will include these cases.")
    
menu = ["Kaplan-Meier Analysis", "Log-rank Results Table"]
choice = st.sidebar.radio("Menu", menu)


### DISPLAY THE FIRST PAGE
### The first page is just a simple Kaplan Meier curve viewer where the use can plot one treatment curve vs. the appropriate control curve
if choice == "Kaplan-Meier Analysis":
    
### Step 1: Create a dropdown menu so the user can select a treatment type of interest, e.g. Rapa: rapamycin, or Met: metformin
# I'd like to show both the abbreviated treatmnt name (in treatment column) and the full name (in full_name column). If it's a combo treatment, we need to show both drugs names.
    
# Function to create a list of values that contain all the treatments in the df, combined with their respective full names. If a row of df has values for treatment2 and full_name2 that indicate it is a combination drug treatment, the second drug abbreviation and name is also added. This list is stored in treatment_fullname. This iterates over all rows of df so there will be repeats
    def create_treatment_fullname(row):
        if pd.isna(row["treatment2"]):
            return row["treatment"] + ": " + row["full_name"]
        else:
            return row["treatment"] + " + " + row["treatment2"] + ": " + row["full_name"] + " + " + row["full_name2"]
    df["treatment_fullname"] = df.apply(create_treatment_fullname, axis=1)
    
# Get unique treatment_fullname values (get rid of all the repeats in treatment_fullname), sort them alphabetically, and display them in a dropdown menu for the user to select
    treatment_fullname_values = sorted(df["treatment_fullname"].unique())
    selected_treatment_fullname = st.sidebar.selectbox("Select a treatment", treatment_fullname_values)

    
### Step 2: Create a second dropdown menu for the user to specify a specific cohort within the treatment of interest they selected. This will show users the details of the different cohorts wihtin the selected treatment of interest, like the dose, age of initiation, etc.
    
# First we need to match the user's selction of treatment(s) back to the other values in df that specify the details of each time that treatment has been tested, since it may have been tested more than once in different years, at different doses, etc. 
    # Custom function to create combo string based that will be displayed in the dropdown menu. This will be called by 'apply' later in the code
    def create_combo_string(row):
        base_string = f"{row['Rx(ppm)']} ppm, {row['age_initiation(mo)']} mo, {row['cohort']}"
        if pd.notna(row["treatment2"]) and pd.notna(row["Rx(ppm)2"]):
            base_string = f"{row['Rx(ppm)']} ppm + {row['Rx(ppm)2']} ppm, {row['age_initiation(mo)']} mo, {row['cohort']}"
        return base_string
    
# Take the user's selection and extract the information needed to match that selection to the appropirate rows of df to get the correct info for cohort, age of initiation, dose, etc.
    # Split the selected_treatment_fullname by ":"
    selected_treatment = selected_treatment_fullname.split(":")[0].strip()

    # If the selected_treatment contains " + " (i.e. it is a drug combination treatment), split by " + " and store those two values, else store just the single treatment
    if " + " in selected_treatment:
        selected_treatments = tuple(selected_treatment.split(" + "))
    else:
        selected_treatments = (selected_treatment,)

    # Filter df based on the selected_treatments, and get the unique combinations of values like treatment, dose, cohort, which define the individual experiments within the same value of treatment
    # Filter the DataFrame based on the selected_treatments.
    if len(selected_treatments) == 1:
        unique_combinations = df[
            (df["treatment"] == selected_treatments[0]) &
            (pd.isna(df["treatment2"]))
        ]
    else:
        unique_combinations = df[
            (df["treatment"] == selected_treatments[0]) &
            (df["treatment2"] == selected_treatments[1])
        ]
    unique_combinations = unique_combinations[["group", "treatment", "treatment2", "Rx(ppm)", "age_initiation(mo)", "cohort", "Rx(ppm)2"]].drop_duplicates()

    # Extract the specific info we want to display in the dropdown menu, like dose, age of initiation, cohort, and the 2nd drug dose, if it exists.
    unique_combinations["combo"] = unique_combinations.apply(create_combo_string, axis=1)

    # Let the user select a unique combination in the dropdown menu
    selected_combo = st.sidebar.selectbox("Select a cohort (dose, age of initiation, and year)", unique_combinations["combo"].tolist())
    selected_rx_ppm, selected_age_initiation, selected_cohort, *rest = selected_combo.split(", ")

### Step 3: Make dropdown menues so the user can further filter by sex and site, if desired
    sex_values = ["m", "f", "m+f"]
    selected_sex = st.sidebar.selectbox("Select sex (m, f, or m+f)", sex_values, index=2)
    
    site_values = ["TJL", "UM", "UT", "TJL+UM", "TJL+UT", "UM+UT", "TJL+UM+UT"]
    selected_site = st.sidebar.selectbox("Select site (TJL, UM, UT, TJL+UM, TJL+UT, UM+UT, or TJL+UM+UT)", site_values, index=6)

### Step 4: Filter df to include only those rows matching the user's selections
    # Again need to treat the cases where there is 1 or 2 treatments differently, here based on the length of the tuple selected_treatments
    if len(selected_treatments) == 1:
        selected_data = df[(df["treatment"] == selected_treatments[0]) & (df["Rx(ppm)"] == selected_rx_ppm) & (df["age_initiation(mo)"] == selected_age_initiation) & (df["cohort"] == selected_cohort)]
    else:
        selected_dose = selected_treatment_fullname.split(":")[1].strip()
        selected_doses = tuple(dose[:-4].strip() for dose in selected_dose.split(" + "))
        selected_data = df[(df["treatment"] == selected_treatments[0]) & (df["Rx(ppm)"] == selected_doses[0]) & (df["treatment2"] == selected_treatment[1]) & (df["Rx(ppm)2"] == selected_doses[1]) & (df["age_initiation(mo)"] == selected_age_initiation) & (df["cohort"] == selected_cohort)]

    # sex and site must be filtered a bit differently since we added
    # Apply the sex filter
    if selected_sex != "m+f":
        selected_data = selected_data[selected_data["sex"] == selected_sex]

    # Apply the site filter
    if selected_site != "TJL+UM+UT":
        selected_sites = selected_site.split("+")
        selected_data = selected_data[selected_data["site"].isin(selected_sites)]

### Step 5: Filter df to include only those rows matching the appropriate control group based on the user's selections
    
    # Filter df to get the control mice corresponding to the cohort the user selected
    control_data = df[(df["treatment"] == "Control") & (df["cohort"] == selected_cohort)]

    # Apply the sex filter to df to match the user selection
    if selected_sex != "m+f":
        control_data = control_data[control_data["sex"] == selected_sex]

    # Apply the site filter to df to match the user selection
    if selected_site != "TJL+UM+UT":
        selected_sites = selected_site.split("+")
        control_data = control_data[control_data["site"].isin(selected_sites)]

    print("Number of NaN values in the selected_data 'age(days)' column:", selected_data["age(days)"].isna().sum())
    print("Number of NaN values in the control_data 'age(days)' column:", control_data["age(days)"].isna().sum())
    
    ### Step 6: Plot the Kaplan Meier curve of the selected treatment vs. the appropriate control
    kmf = KaplanMeierFitter()

    # Convert dead column to integer values (1 for True, 0 for False)
    selected_data["dead"] = selected_data["dead"].astype(int)
    control_data["dead"] = control_data["dead"].astype(int)

    # Remove rows with non-numeric values in the "age(days)" column
    selected_data = selected_data[pd.to_numeric(selected_data["age(days)"], errors='coerce').notna()]
    control_data = control_data[pd.to_numeric(control_data["age(days)"], errors='coerce').notna()]

    # Convert age(days) column to numeric values
    selected_data["age(days)"] = pd.to_numeric(selected_data["age(days)"], errors='coerce')
    control_data["age(days)"] = pd.to_numeric(control_data["age(days)"], errors='coerce')

    kmf.fit(selected_data["age(days)"], event_observed=selected_data["dead"], label=selected_treatment)
    kmf_control = KaplanMeierFitter()
    kmf_control.fit(control_data["age(days)"], event_observed=control_data["dead"], label="Control")

 













# DISPLAY THE SECOND PAGE
elif choice == "Log-rank Results Table":
    st.title("Logrank Results")
    st.write("This table displays statistics and results for all ITP-tested aging interventions compared to the matching control group. It not only contains the analysis for e.g. pooled data across each of the testing sites for any particular treatment, but also for e.g. each site individually or any combination of 2 sites. Same for sex: m only, f only, or combined.")
    st.write("The table is sortable on any column, and you can filter based on the values of any column in the lefthand sidebar.")

    # Filters
    # Add a new column that combines treatment and full_name values
    # Custom function to create treatment_fullname based on the presence of treatment2 (which is most often empty when there is only a single treatment)
    def create_treatment_fullname(row):
        if pd.isna(row["treatment2"]):
            return row["treatment"] + ": " + row["full_name"]
        else:
            return row["treatment"] + " + " + row["treatment2"] + ": " + row["full_name"] + " + " + row["full_name2"]

    # Use apply with the custom function to create the treatment_fullname column
    logrank_df["treatment_fullname"] = logrank_df.apply(create_treatment_fullname, axis=1)

    # Get unique treatment_fullname values
    treatment_fullname_values = sorted(logrank_df["treatment_fullname"].unique())
    selected_treatment_fullname = st.sidebar.multiselect("Filter by treatment", treatment_fullname_values, default=treatment_fullname_values, key="treatment_filter")  
          
    sex_values = ["m", "f", "m+f"]
    selected_sex_filter = st.sidebar.multiselect("Filter by sex", sex_values, default=["m", "f"])

    site_values = ["TJL", "UM", "UT", "TJL+UM", "TJL+UT", "UM+UT", "TJL+UM+UT"]
    selected_site_filter = st.sidebar.multiselect("Filter by site", site_values, default=["TJL+UM+UT"])
    
    rx_ppm_values = sorted(logrank_df["Rx(ppm)"].unique())
    selected_rx_ppm = st.sidebar.multiselect("Filter by dose (ppm)", rx_ppm_values, default=rx_ppm_values)

    age_initiation_values = sorted(logrank_df["age_initiation(mo)"].unique())
    selected_age_initiation = st.sidebar.multiselect("Filter by age of initiation (mo)", age_initiation_values, default=age_initiation_values)

    cohorts = sorted(logrank_df["cohort"].unique())
    selected_cohort_filter = st.sidebar.multiselect("Filter by cohort", cohorts, default=cohorts)

    filtered_logrank_df = logrank_df[
        logrank_df["treatment_fullname"].isin(selected_treatment_fullname) &
        logrank_df["sex"].isin(selected_sex_filter) &
        logrank_df["site"].isin(selected_site_filter) &
        logrank_df["Rx(ppm)"].isin(selected_rx_ppm) &
        logrank_df["age_initiation(mo)"].isin(selected_age_initiation) &
        logrank_df["cohort"].isin(selected_cohort_filter)
    ]

        
    # formatting for values in the table, like remove decimal places
    filtered_logrank_df['%_lifespan_increase'] = filtered_logrank_df['%_lifespan_increase'].round(1)
    filtered_logrank_df['test_statistic'] = filtered_logrank_df['test_statistic'].round(1)
    # Convert p-value to scientific notation if the value is < 0.001
    filtered_logrank_df['p-value'] = filtered_logrank_df['p-value'].apply(lambda x: f'{x:.1e}' if x < 0.001 else f'{x:.3f}')
        
    # Exclude the specific columns
    filtered_logrank_df_display = filtered_logrank_df.drop(columns=['group', 'full_name', 'treatment_fullname'])
    # Display the table without the excluded columns
    st.write(filtered_logrank_df_display.reset_index().drop(columns=['index']))


# If the user chooses an invalid option, display an error message
else:
    st.error("Invalid selection. Please choose a valid option from the sidebar.")


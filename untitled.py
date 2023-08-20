###Temp trying new power

    def bootstrap_median_difference(data, sample_size):
        sample = data.sample(n=sample_size, replace=True)
        kmf = KaplanMeierFitter()
        control_data = sample[sample['group'] == 'Control']
        kmf.fit(control_data['age(days)'], event_observed=control_data['dead'])
        median_control = kmf.median_survival_time_
        hypothetical_treatment_median = median_control * 1.20  # 20% increase
        return median_control, hypothetical_treatment_median

    # Slider for bootstrap sample size
    bootstrap_sample_size = st.sidebar.slider('Bootstrap Sample Size', min_value=10, max_value=len(data), value=len(data))

    # Number of bootstrap iterations
    n_iterations = 100
    bootstrap_results = [bootstrap_median_difference(data, bootstrap_sample_size) for _ in range(n_iterations)]

    # Compute power: proportion of times the difference in median survival times 
    # between the bootstrapped control and treatment groups exceeds a threshold
    threshold = st.sidebar.slider('Set a threshold for significant median lifespan difference', 5, 50, 10)  # Let users choose a threshold
    diffs = [treatment - control for control, treatment in bootstrap_results]
    power = np.mean([diff > threshold for diff in diffs])

    st.write(f"Estimated power based on bootstrapping: {power:.4f}")



import json

# Load JSON data from file
with open('output_data_entropy_plots/COPY_BEFORE_MODIFICATIONS_Run_8_in_word_doc_output_data_entropy_plot_budget_1000000_and_100000_samples_per_bin_and_NUM_BINS_20.json', 'r') as file:
    data = json.load(file)

# Modify the relevant data in bins_at_end
bins_data = data['bins_at_end']
for bin_range, values in bins_data.items():
    # Multiply each value in 'expected_values_detected' by 0.01
    values['expected_values_detected'] = [value * 0.01 for value in values['expected_values_detected']]
    # Multiply each value in 'expected_total_values' by 0.01
    values['expected_total_values'] = [value * 0.01 for value in values['expected_total_values']]
    # Multiply 'final_expected_value_detected' by 0.01
    values['final_expected_value_detected'] *= 0.01

# Optional: Save the modified data back to a file
with open('modified_data.json', 'w') as file:
    json.dump(data, file, indent=4)

# Print the modified data (or just confirm modification)
print('Data modification complete.')

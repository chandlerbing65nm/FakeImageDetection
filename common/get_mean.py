def calculate_averages(input_data):
    # Split the input data into lines and remove the header
    lines = input_data.strip().split("\n")
    data_lines = lines[1:]
    
    # Parse the data into a dictionary: model name -> [Avg.Prec., Acc., AUC]
    data = {}
    for line in data_lines:
        parts = line.split(", ")
        model_name = parts[0]
        # Convert the metrics to float
        values = list(map(float, parts[1:]))
        data[model_name] = values

    # Define the two groups.
    # Group 1: GAN-based models
    group1_models = [
        "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN",
        "CycleGAN", "StarGAN", "GauGAN", "DeepFake"
    ]
    # Group 2: Diffusion/Guided models
    group2_models = [
        "Guided", "LDM_200", "LDM_200_cfg", "LDM_100",
        "Glide_100_27", "Glide_50_27", "Glide_100_10", "DALL-E"
    ]
    
    # Collect the rows for each group
    group1_values = []
    for model in group1_models:
        if model in data:
            group1_values.append(data[model])
        else:
            print(f"Warning: {model} not found in the input data.")
    
    group2_values = []
    for model in group2_models:
        if model in data:
            group2_values.append(data[model])
        else:
            print(f"Warning: {model} not found in the input data.")
    
    # Function to compute averages for each column from a list of rows
    def compute_average(rows):
        n = len(rows)
        return [sum(col) / n for col in zip(*rows)]
    
    avg_group1 = compute_average(group1_values) if group1_values else [0, 0, 0]
    avg_group2 = compute_average(group2_values) if group2_values else [0, 0, 0]
    
    results = {
        "Group 1 (GAN Models)": avg_group1,
        "Group 2 (Diffusion/Guided Models)": avg_group2
    }
    return results


# Example input
input_data = """Dataset, Avg.Prec., Acc., AUC
ProGAN, 100.00, 99.64, 1.000
CycleGAN, 89.87, 79.64, 0.883
BigGAN, 89.66, 73.40, 0.891
StyleGAN, 94.99, 81.16, 0.946
GauGAN, 96.28, 84.85, 0.965
StarGAN, 72.60, 65.56, 0.617
DeepFake, 75.17, 51.66, 0.764
SITD, 82.00, 73.89, 0.839
SAN, 73.31, 52.74, 0.751
CRN, 96.50, 87.71, 0.966
IMLE, 93.63, 79.99, 0.945
Guided, 65.46, 63.50, 0.685
LDM_200, 96.02, 88.80, 0.960
LDM_200_cfg, 95.90, 88.60, 0.958
LDM_100, 96.14, 89.05, 0.960
Glide_100_27, 80.56, 68.65, 0.875
Glide_50_27, 80.86, 69.95, 0.882
Glide_100_10, 83.36, 73.20, 0.908
DALL-E, 95.24, 87.60, 0.956
"""

# Calculate and print the results
averages = calculate_averages(input_data)
for group, values in averages.items():
    print(f"{group} Averages:")
    print(f"  Avg.Prec.: {values[0]:.2f}")
    print(f"  Acc.: {values[1]:.2f}")
    print(f"  AUC: {values[2]:.3f}\n")
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
ProGAN, 100.00, 99.94, 1.000
CycleGAN, 90.49, 74.75, 0.918
BigGAN, 84.55, 78.10, 0.867
StyleGAN, 99.61, 96.39, 0.995
StyleGAN2, 99.61, 96.39, 0.995
GauGAN, 77.76, 70.90, 0.826
StarGAN, 100.00, 98.10, 1.000
DeepFake, 90.15, 50.25, 0.921
SITD, 91.21, 83.89, 0.920
SAN, 87.34, 82.88, 0.876
CRN, 67.59, 50.00, 0.760
IMLE, 67.72, 50.00, 0.761
Guided, 94.81, 86.05, 0.948
LDM_200, 98.39, 93.80, 0.979
LDM_200_cfg, 96.94, 91.15, 0.961
LDM_100, 98.56, 93.60, 0.982
Glide_100_27, 88.65, 79.40, 0.851
Glide_50_27, 92.66, 84.35, 0.905
Glide_100_10, 91.22, 82.70, 0.886
DALL-E, 93.58, 85.45, 0.918
"""

# Calculate and print the results
averages = calculate_averages(input_data)
for group, values in averages.items():
    print(f"{group} Averages:")
    print(f"  Avg.Prec.: {values[0]:.2f}")
    print(f"  Acc.: {values[1]:.2f}")
    print(f"  AUC: {values[2]:.3f}\n")
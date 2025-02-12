def calculate_averages(input_data):
    # Split the input data into lines
    lines = input_data.strip().split("\n")
    
    # Skip the header lines
    data_lines = lines[1:]
    
    # Initialize lists to store the values
    avg_prec_values = []
    acc_values = []
    auc_values = []
    
    # Function to compute the average for given rows
    def compute_average(rows):
        return [sum(column) / len(column) for column in zip(*rows)]
    
    # Parse the input data into structured format
    data = []
    for line in data_lines:
        parts = line.split(", ")
        data.append([parts[0]] + list(map(float, parts[1:])))
    
    # Define row ranges for specific categories
    categories = {
        "GANs": range(0, 6),
        "DeepFake": [6],
        "Low-level vision": range(7, 9),
        "Perceptual loss": range(9, 11),
        "Guided diffusion": [11],
        "Latent diffusion": range(12, 15),
        "Glide": range(15, 18),
        "DALL-E": [18]
    }
    
    # Extract and calculate averages for each category
    results = {}
    for category, rows in categories.items():
        selected_rows = [data[i][1:] for i in rows]  # Extract relevant rows (skip the name)
        if len(selected_rows) > 1:
            results[category] = compute_average(selected_rows)
        else:
            results[category] = selected_rows[0]
    
    # Calculate overall averages for columns 2, 3, 4
    all_values = [row[1:] for row in data]  # Skip the name column
    overall_avg = compute_average(all_values)
    
    # Format and return results
    results["Overall"] = overall_avg
    return results

# Example input
input_data = """Dataset, Avg.Prec., Acc., AUC
ProGAN, 100.00, 99.94, 1.000
CycleGAN, 90.49, 74.75, 0.918
BigGAN, 84.55, 78.10, 0.867
StyleGAN, 99.61, 96.39, 0.995
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


# Calculate and print results
averages = calculate_averages(input_data)
for category, values in averages.items():
    print(f"{category} Averages:\nAvg.Prec.: {values[0]:.2f}\nAcc.: {values[1]:.2f}\nAUC: {values[2]:.3f}")

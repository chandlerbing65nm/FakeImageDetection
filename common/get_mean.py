def calculate_averages(input_data):
    # Split the input data into lines
    lines = input_data.strip().split("\n")
    
    # Skip the header lines
    data_lines = lines[2:]
    
    # Initialize lists to store the values
    avg_prec_values = []
    acc_values = []
    auc_values = []
    
    # Parse each line of data
    for line in data_lines:
        parts = line.split(", ")
        avg_prec_values.append(float(parts[1]))
        acc_values.append(float(parts[2]))
        auc_values.append(float(parts[3]))
    
    # Calculate averages
    avg_prec_avg = sum(avg_prec_values) / len(avg_prec_values)
    acc_avg = sum(acc_values) / len(acc_values)
    auc_avg = sum(auc_values) / len(auc_values)
    
    # Return the averages
    return {
        "Avg.Prec.": avg_prec_avg,
        "Acc.": acc_avg,
        "AUC": auc_avg
    }

# Example input
input_data = """Dataset, Avg.Prec., Acc., AUC
----------------------------
ProGAN, 100.00, 99.95, 1.000
CycleGAN, 87.85, 78.46, 0.890
BigGAN, 81.02, 76.80, 0.849
StyleGAN, 98.58, 89.78, 0.983
GauGAN, 78.56, 71.44, 0.825
StarGAN, 99.80, 98.05, 0.997
DeepFake, 92.16, 69.40, 0.928
SITD, 96.38, 74.72, 0.964
SAN, 65.26, 58.45, 0.678
CRN, 73.32, 53.01, 0.779
IMLE, 82.67, 53.02, 0.885
Guided, 82.55, 74.30, 0.818
LDM_200, 92.83, 80.55, 0.919
LDM_200_cfg, 90.49, 77.15, 0.892
LDM_100, 93.05, 81.20, 0.920
Glide_100_27, 88.58, 73.10, 0.877
Glide_50_27, 91.70, 78.35, 0.911
Glide_100_10, 89.21, 74.05, 0.886
DALL-E, 79.65, 65.90, 0.767
"""

# Calculate and print averages
averages = calculate_averages(input_data)
print(f"Averages:\nAvg.Prec.: {averages['Avg.Prec.']:.2f}\nAcc.: {averages['Acc.']:.2f}\nAUC: {averages['AUC']:.3f}")

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
ProGAN, 100.00, 100.00, 1.000
CycleGAN, 83.25, 80.17, 0.872
BigGAN, 94.30, 88.48, 0.955
StyleGAN, 99.95, 93.00, 0.999
GauGAN, 91.70, 86.17, 0.940
StarGAN, 99.99, 99.45, 1.000
DeepFake, 91.24, 67.77, 0.892
SITD, 92.77, 83.33, 0.947
SAN, 73.69, 55.71, 0.743
CRN, 98.19, 65.58, 0.989
IMLE, 97.85, 65.58, 0.985
Guided, 78.29, 70.45, 0.798
LDM_200, 87.01, 64.15, 0.873
LDM_200_cfg, 86.86, 64.45, 0.867
LDM_100, 87.95, 65.35, 0.882
Glide_100_27, 88.24, 66.90, 0.879
Glide_50_27, 92.50, 72.90, 0.922
Glide_100_10, 89.33, 68.05, 0.893
DALL-E, 91.32, 68.00, 0.918
"""

# Calculate and print averages
averages = calculate_averages(input_data)
print(f"Averages:\nAvg.Prec.: {averages['Avg.Prec.']:.2f}\nAcc.: {averages['Acc.']:.2f}\nAUC: {averages['AUC']:.3f}")

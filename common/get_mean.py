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
ProGAN, 99.86, 96.97, 0.999
CycleGAN, 97.91, 88.61, 0.978
BigGAN, 94.11, 79.00, 0.934
StyleGAN, 91.36, 69.70, 0.900
GauGAN, 99.70, 96.34, 0.997
StarGAN, 96.96, 86.19, 0.962
DeepFake, 75.68, 54.41, 0.737
SITD, 65.30, 60.56, 0.633
SAN, 71.76, 51.14, 0.731
CRN, 79.64, 51.94, 0.781
IMLE, 96.34, 62.75, 0.960
Guided, 82.36, 63.55, 0.825
LDM_200, 93.12, 71.35, 0.930
LDM_200_cfg, 76.46, 55.10, 0.771
LDM_100, 93.51, 72.85, 0.932
Glide_100_27, 89.28, 65.90, 0.889
Glide_50_27, 91.19, 67.65, 0.909
Glide_100_10, 89.71, 66.40, 0.894
DALL-E, 78.37, 57.25, 0.772
"""

# Calculate and print averages
averages = calculate_averages(input_data)
print(f"Averages:\nAvg.Prec.: {averages['Avg.Prec.']:.2f}\nAcc.: {averages['Acc.']:.2f}\nAUC: {averages['AUC']:.3f}")

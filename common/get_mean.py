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
ProGAN, 100.00, 99.99, 1.000
CycleGAN, 91.79, 82.55, 0.927
BigGAN, 82.28, 69.10, 0.869
StyleGAN, 99.52, 82.74, 0.995
GauGAN, 91.62, 83.67, 0.933
StarGAN, 99.11, 89.94, 0.990
DeepFake, 90.75, 53.91, 0.903
SITD, 92.24, 88.61, 0.940
SAN, 72.51, 51.14, 0.747
CRN, 97.41, 87.28, 0.980
IMLE, 98.25, 87.38, 0.987
Guided, 78.44, 62.75, 0.795
LDM_200, 78.80, 56.50, 0.786
LDM_200_cfg, 77.19, 57.00, 0.760
LDM_100, 79.50, 57.20, 0.792
Glide_100_27, 84.92, 63.25, 0.836
Glide_50_27, 88.58, 66.75, 0.872
Glide_100_10, 86.28, 64.55, 0.848
DALL-E, 70.84, 57.45, 0.659
"""

# Calculate and print averages
averages = calculate_averages(input_data)
print(f"Averages:\nAvg.Prec.: {averages['Avg.Prec.']:.2f}\nAcc.: {averages['Acc.']:.2f}\nAUC: {averages['AUC']:.3f}")

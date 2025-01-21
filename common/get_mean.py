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
Guided, 89.25, 81.00, 0.898
LDM_200, 95.53, 89.65, 0.936
LDM_200_cfg, 91.71, 84.20, 0.883
LDM_100, 96.36, 90.25, 0.950
Glide_100_27, 75.23, 68.50, 0.669
Glide_50_27, 78.85, 69.75, 0.730
Glide_100_10, 79.51, 70.25, 0.743
DALL-E, 80.43, 71.40, 0.749
"""

# Calculate and print averages
averages = calculate_averages(input_data)
print(f"Averages:\nAvg.Prec.: {averages['Avg.Prec.']:.2f}\nAcc.: {averages['Acc.']:.2f}\nAUC: {averages['AUC']:.3f}")

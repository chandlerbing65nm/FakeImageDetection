import matplotlib.pyplot as plt
import numpy as np

# from Wang et al. progan
H_i = [
    0.495, 0.301, 0.093, 0.139, 0.038, 0.028, 0.028, 0.060, 0.042, 0.093, 0.330, 0.081, 
    0.054, 0.129, 0.181, 0.169, 0.086, 0.053, 0.124, 0.014, 0.012, 0.024, 0.076, 0.317, 
    0.471, 0.012, 0.022, 0.008, 0.001, 0.001, 0.000, 0.002, 0.002, 0.003, 0.000, 0.000, 
    0.000, 0.003, 0.001, 0.002, 0.016, 0.019, 0.024, 0.001, 0.002, 0.001, 0.003, 0.003, 0.003
]
L_i = [
    0.413, 0.320, 0.179, 0.217, 0.118, 0.071, 0.082, 0.127, 0.085, 0.151, 0.310, 0.174, 
    0.079, 0.149, 0.187, 0.167, 0.131, 0.097, 0.163, 0.037, 0.045, 0.062, 0.133, 0.300, 
    0.378, 0.035, 0.077, 0.028, 0.000, 0.006, 0.001, 0.007, 0.003, 0.013, 0.001, 0.002, 
    0.001, 0.002, 0.003, 0.003, 0.036, 0.057, 0.054, 0.000, 0.006, -0.005, 0.012, 0.010, 0.011
]

# # from Gragnaniello et al. progan
# H_i = [
#     0.472, 0.018, 0.155, 0.071, 0.003, 0.003, 0.003, 0.002, 0.002, 0.004, 0.055, 0.033, 
#     0.048, 0.352, 0.437, 0.149, 0.138, 0.085, 0.070, 0.050, 0.000, 0.000, 0.154, 0.309, 
#     0.181, 0.015, 0.264, 0.051, 0.018, 0.026, 0.015, 0.029, 0.034, 0.017, 0.000, 0.002, 
#     0.000, 0.000, 0.000, 0.012, 0.146, 0.133, 0.011, 0.001, 0.015, 0.009, 0.000, 0.002, 0.000
# ]
# L_i = [
#     0.409, 0.079, 0.208, 0.101, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.127, 0.121, 
#     0.130, 0.352, 0.397, 0.191, 0.175, 0.152, 0.103, 0.106, 0.000, 0.000, 0.228, 0.355, 
#     0.246, 0.050, 0.284, 0.110, 0.046, 0.065, 0.040, 0.041, 0.047, 0.034, 0.000, 0.000, 
#     0.000, 0.001, 0.000, 0.027, 0.192, 0.169, 0.044, 0.003, 0.054, 0.022, 0.000, 0.002, 0.000
# ]

# Stage information for x-axis reference
stages =['0'] * 1 + ['1'] * 9 + ['2'] * 12 + ['3'] * 18 + ['4'] * 9
x = np.arange(len(H_i))  # Points

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, H_i, label="High Freq Masking", marker='o')
plt.plot(x, L_i, label="Low Freq Masking", marker='x')

# Vertical dashed lines at the start of each group
start_of_groups = [1, 10, 22, 39]  # Calculated based on the number of points in each group
for start in start_of_groups:
    plt.axvline(x=start, color='gray', linestyle='--', linewidth=1)


# Customizing the plot
plt.title("mAP Sensitivity of Layer-wise Pruning")
plt.xlabel("ResNet-50 Stages")
plt.ylabel("Sensitivity")
plt.xticks(x, labels=stages, rotation=0, ha='right')
plt.tight_layout()
plt.legend()

# Save the plot
file_path = 'pruning_sensitivity-Wang.png'
plt.savefig(file_path)
plt.close()  # Close the plot to prevent it from displaying in the output

file_path

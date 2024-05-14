import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Define the base directory for the .mat files
base_dir = './output_files/curves/clip_rn50/maskratio70/clip_rn50_ndz_0004_rdcurves'

# Initialize lists to store x and y values
x_values = []
y_values = []

# Define the alpha for EMA
alpha = 0.1

# Initialize the EMA variable
ema = None

# Store unique iterations and layers for determining x-axis labels
iterations_set = set()
layers_set = set()

# Iterate through layers and iterations
for iteration in range(1, 2):
    for layer in range(49):
        # Format the file path
        file_path = os.path.join(base_dir, f'rnd#{iteration}', f'clip_rn50_{layer:03d}.mat')
        
        # Check if the file exists
        if os.path.isfile(file_path):
            # Load the .mat file
            mat_data = scipy.io.loadmat(file_path)
            
            # Extract the numpy array and get the value at the 4th position
            array = mat_data['rd_dist'][0]  # Access the array correctly
            value = array[4]  # Extract the value at the 4th position
            
            # Calculate the EMA
            if ema is None:
                ema = value
            else:
                ema = alpha * value + (1 - alpha) * ema
            
            # Append the x and y values
            x_values.append(f'{layer:02d}-{iteration:02d}')
            y_values.append(ema)
            
            # Add to the sets for determining x-axis labels
            iterations_set.add(iteration)
            layers_set.add(layer)

# Determine the x-axis labels
if len(iterations_set) == 1:
    x_labels = [x.split('-')[0] for x in x_values]  # Use only layers if one iteration
else:
    x_labels = [x for x in x_values]  # Use only iterations if more than one iteration

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(x_labels, y_values, marker='o', linestyle='-')
plt.xlabel('Model Layers' if len(iterations_set) == 1 else 'Iterations')
plt.ylabel('Exponential Moving Average')
plt.title('LayerWise Sensitivity to Pruning')
plt.xticks(rotation=45)
plt.tight_layout()


# Save the figure
os.makedirs(f"./output_files/plots/", exist_ok=True)
plt.savefig('./output_files/plots/rd_plot.png')

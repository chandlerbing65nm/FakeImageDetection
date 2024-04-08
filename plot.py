import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Stage": ["1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3", "3", "3", "4", "4", "4"],
    "Conv Block": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    "ImageNet": [0.429, 0.154, 0.381, 0.019, 0.059, 0.026, 0.086, 0.015, 0.016, 0.017, 0.017, 0.015, 0.012, 0.217, 0.564, 2.744],
    "ProGAN": [0.881, 0.312, 0.885, 0.042, 0.120, 0.062, 0.219, 0.034, 0.036, 0.036, 0.036, 0.032, 0.024, 0.437, 1.138, 5.223],
    "Glide": [0.177, 0.097, 0.262, 0.011, 0.032, 0.016, 0.062, 0.008, 0.009, 0.009, 0.009, 0.008, 0.006, 0.095, 0.228, 1.035]
}

# Creating DataFrame
df = pd.DataFrame(data)
df["Stage-Block"] = df["Stage"] + "-" + df["Conv Block"].astype(str)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df["Stage-Block"], df["ImageNet"], marker='o', linestyle='-', color='blue', label='ImageNet')
plt.plot(df["Stage-Block"], df["ProGAN"], marker='x', linestyle='-', color='red', label='ProGAN')
plt.plot(df["Stage-Block"], df["DALL-E"], marker='*', linestyle='-', color='green', label='DALL-E')

# Adding a horizontal dashed line at the y-coordinate for "Stage-Block" 1-3 in ProGAN
# y_coord_forensynths = df[df["Stage-Block"] == "2-1"]["ProGAN"].values[0]
# plt.axhline(y=y_coord_forensynths, color='gray', linestyle='--', linewidth=1)

plt.title("Spectral Energy after DCT")
plt.xlabel("ResNet-50 Stage-Block")
plt.ylabel("Spectral Energy")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Show the plot
plt.savefig(f'spectral_energy_highband.png')

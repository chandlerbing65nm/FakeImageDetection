import pandas as pd

# Input data
data = """
ProGAN, 100.00, 100.00, 1.000
CycleGAN, 94.07, 85.28, 0.937
BigGAN, 97.61, 92.35, 0.980
StyleGAN, 99.92, 90.76, 0.999
GauGAN, 98.54, 93.80, 0.987
StarGAN, 99.99, 99.57, 1.000
DeepFake, 94.00, 63.57, 0.930
SITD, 95.46, 88.89, 0.957
SAN, 81.47, 55.48, 0.818
CRN, 96.72, 90.51, 0.973
IMLE, 95.47, 89.31, 0.962
Guided, 80.14, 63.30, 0.795
LDM_200, 94.35, 66.25, 0.942
LDM_200_cfg, 94.24, 68.05, 0.938
LDM_100, 94.77, 65.50, 0.948
Glide_100_27, 90.32, 60.90, 0.900
Glide_50_27, 93.46, 64.25, 0.931
Glide_100_10, 91.05, 60.45, 0.909
DALL-E, 96.29, 67.60, 0.964
"""

# Process the data
df = pd.DataFrame([line.split(", ") for line in data.strip().split("\n")], columns=["Model", "Col2", "Col3", "Col4"])
df["Col2"] = df["Col2"].astype(float)

# Get the horizontal values from column 2, separated by '&'
horizontal_values = ' & '.join(df["Col2"].astype(str))
print(horizontal_values)

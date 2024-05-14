import matplotlib.pyplot as plt
import numpy as np

# Example data lists (fill in your actual data where the ellipses are)
L_a = [
    0.413, 0.320, 0.179, 0.217, 0.118, 0.071, 0.082, 0.127, 0.085, 0.151, 0.310, 0.174, 
    0.079, 0.149, 0.187, 0.167, 0.131, 0.097, 0.163, 0.037, 0.045, 0.062, 0.133, 0.300, 
    0.378, 0.035, 0.077, 0.028, 0.000, 0.006, 0.001, 0.007, 0.003, 0.013, 0.001, 0.002, 
    0.001, 0.002, 0.003, 0.003, 0.036, 0.057, 0.054, 0.000, 0.006, -0.005, 0.012, 0.010, 0.011
]
L_b = [
    0.409, 0.079, 0.208, 0.101, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.127, 0.121, 
    0.130, 0.352, 0.397, 0.191, 0.175, 0.152, 0.103, 0.106, 0.000, 0.000, 0.228, 0.355, 
    0.246, 0.050, 0.284, 0.110, 0.046, 0.065, 0.040, 0.041, 0.047, 0.034, 0.000, 0.000, 
    0.000, 0.001, 0.000, 0.027, 0.192, 0.169, 0.044, 0.003, 0.054, 0.022, 0.000, 0.002, 0.000
]
L_c = [
    0.350, 0.333, 0.035, 0.263, 0.136, 0.072, 0.337, 0.000, 0.054, 0.185, 
    0.343, 0.243, 0.289, 0.004, 0.000, 0.000, 0.058, 0.034, 0.000, 0.026, 0.000, 0.024, 
    0.289, 0.008, 0.023, 0.017, 0.005, 0.013, 0.000, 0.009, 0.000, 0.000, 0.000, 0.000, 
    0.000, 0.000, 0.013, 0.010, 0.000, 0.003, 0.068, 0.000, 0.000, 0.002, 0.004, 0.001, 0.019, 0.003, 0.006
]
L_d = [
    0.000, 0.112, 0.133, 0.182, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.311, 0.217, 
    0.120, 0.152, 0.000, 0.190, 0.176, 0.177, 0.186, 0.026, 0.000, 0.000, 0.261, 0.078, 
    0.052, 0.027, 0.042, 0.031, 0.014, 0.000, 0.022, 0.004, 0.000, 0.007, 0.005, 0.003, 
    0.002, 0.002, 0.000, 0.003, 0.020, 0.000, 0.064, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000
]
L_e = [
    0.107, 0.216, 0.066, 0.177, 0.000, 0.039, 0.206, 0.000, 0.000, 0.000, 
    0.151, 0.035, 0.000, 0.000, 0.006, 0.000, 0.000, 0.132, 0.010, 0.034, 0.047, 0.033, 
    0.000, 0.048, 0.036, 0.010, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 
    0.000, 0.000, 0.003, 0.000, 0.003, 0.000, 0.000, 0.006, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
]


L_f = [
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.015, 
    0.000, 0.000, 0.004, 0.000, 0.000, 0.012, 0.000, 0.000, 0.000, 0.000, 0.010, 0.008, 
    0.010, 0.013, 0.012, 0.005, 0.008, 0.000, 0.000, 0.009, 0.000, 0.000, 0.000, 0.006, 
    0.000, 0.024, 0.000, 0.000, 0.026, 0.000, 0.018, 0.000, 0.007, 0.001, 0.033, 0.012, 0.016
]

L_g = [
    0.007, 0.000, 0.041, 0.001, 0.000, 0.000, 0.009, 0.000, 0.000, 0.000, 
    0.000, 0.016, 0.002, 0.010, 0.036, 0.020, 0.013, 0.004, 0.027, 0.006, 0.000, 0.000, 
    0.016, 0.013, 0.000, 0.020, 0.000, 0.015, 0.015, 0.032, 0.021, 0.012, 0.057, 0.010, 
    0.000, 0.012, 0.028, 0.000, 0.000, 0.040, 0.031, 0.020, 0.018, 0.000, 0.011, 0.000, 0.007, 0.068, 0.017
]


# CIFAR10
L_h = [
    0.324, 0.000, 0.030, 0.050, 0.002, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
    0.000, 0.000, 0.000, 0.000, 0.028, 0.015, 0.001, 0.014, 0.058, 0.099, 0.042, 0.080, 0.000, 
    0.000, 0.034, 0.000, 0.000, 0.000, 0.010, 0.001, 0.003, 0.023, 0.000, 0.000, 0.101, 0.000, 
    0.011, 0.022, 0.000, 0.011, 0.003, 0.000, 0.000, 0.009, 0.004, 0.000
]

L_i = [
    0.093, 0.000, 0.000, 0.196, 0.000, 0.201, 0.000, 0.000, 0.000, 0.000, 0.004, 0.000, 0.000, 
    0.000, 0.015, 0.008
]


# Function to calculate the exponential moving average
def exponential_moving_average(data, alpha):
    ema = [data[0]]
    for x in data[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema

# Alpha value for EMA (the smoothing factor, typically between 0.1 and 0.3)
alpha = 0.1

# Calculate exponential moving averages
ma_L_a = exponential_moving_average(L_a, alpha)
ma_L_b = exponential_moving_average(L_b, alpha)
ma_L_c = exponential_moving_average(L_c, alpha)
ma_L_d = exponential_moving_average(L_d, alpha)
ma_L_e = exponential_moving_average(L_e, alpha)
ma_L_f = exponential_moving_average(L_f, alpha)
ma_L_g = exponential_moving_average(L_g, alpha)
ma_L_h = exponential_moving_average(L_h, alpha)
ma_L_i = exponential_moving_average(L_i, alpha)

# Normalized x values
x_L_a = np.linspace(0, 1, len(ma_L_a))
x_L_b = np.linspace(0, 1, len(ma_L_b))
x_L_c = np.linspace(0, 1, len(ma_L_c))
x_L_d = np.linspace(0, 1, len(ma_L_d))
x_L_e = np.linspace(0, 1, len(ma_L_e))
x_L_f = np.linspace(0, 1, len(ma_L_f))
x_L_g = np.linspace(0, 1, len(ma_L_g))
x_L_h = np.linspace(0, 1, len(ma_L_h))
x_L_i = np.linspace(0, 1, len(ma_L_i))

# Create a line plot
plt.figure()
# plt.plot(x_L_a, ma_L_a, label='DFweights-ProGAN-RN50')
# plt.plot(x_L_b, ma_L_b, label='DFweights-ProGAN-modifiedRN50')
# plt.plot(x_L_c, ma_L_c, label='DFweights-ProGAN-CLIPRN50')
# plt.plot(x_L_d, ma_L_d, label='DFweights-LSUN(zebra-vs-apple)-RN50')
# plt.plot(x_L_e, ma_L_e, label='DFweights-LSUN(zebra-vs-apple)-CLIPRN50')
# plt.plot(x_L_f, ma_L_f, label='INweights-ProGAN-RN50')
# plt.plot(x_L_g, ma_L_g, label='INweights-ProGAN-CLIPRN50')
# plt.plot(x_L_h, ma_L_h, label='CFweights-CIFAR10-RN50')
# plt.plot(x_L_i, ma_L_i, label='CFweights-CIFAR10-VGG')

plt.plot(ma_L_a, label='DFweights-ProGAN-RN50')
plt.plot(ma_L_b, label='DFweights-ProGAN-modifiedRN50')
plt.plot(ma_L_c, label='DFweights-ProGAN-CLIPRN50')
plt.plot(ma_L_d, label='DFweights-LSUN(zebra-vs-apple)-RN50')
plt.plot(ma_L_e, label='DFweights-LSUN(zebra-vs-apple)-CLIPRN50')
plt.plot(ma_L_f, label='INweights-ProGAN-RN50')
plt.plot(ma_L_g, label='INweights-ProGAN-CLIPRN50')
# plt.plot(x_L_h, ma_L_h, label='CFweights-CIFAR10-RN50')
# plt.plot(x_L_i, ma_L_i, label='CFweights-CIFAR10-VGG')

plt.xlabel('Model Layers')
plt.ylabel('Sensitivity')
plt.title('Moving Average of Sensitivity')
plt.legend()

# Save the figure as a JPEG file
plt.savefig('pruning_comparison.jpg', format='jpg')
plt.show()


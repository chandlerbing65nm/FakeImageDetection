def calculate_conv_params(out_channels, in_channels, kernel_size, pruning_ratio=0.0):
    pruned_out_channels = int(out_channels * (1 - pruning_ratio))
    # The number of parameters is the product of in_channels, pruned_out_channels, and kernel size squared
    return pruned_out_channels * in_channels * kernel_size * kernel_size

def calculate_fc_params(out_features, in_features, pruning_ratio=0.0):
    pruned_in_features = int(in_features * (1 - pruning_ratio))
    # The number of parameters is the product of pruned_in_features and out_features
    return pruned_in_features * out_features

def calculate_bottleneck_params(in_channels, mid_channels, out_channels, pruning_ratio=0.0):
    mid_channels_pruned = int(mid_channels * (1 - pruning_ratio))
    out_channels_pruned = int(out_channels * (1 - pruning_ratio))

    conv1_params = calculate_conv_params(mid_channels_pruned, in_channels, 1)
    conv2_params = calculate_conv_params(mid_channels_pruned, mid_channels_pruned, 3)
    conv3_params = calculate_conv_params(out_channels_pruned, mid_channels_pruned, 1)

    return conv1_params + conv2_params + conv3_params

def calculate_resnet50_params(pruning_ratio=0.0):
    conv1_out_channels = 64
    total_params = calculate_conv_params(conv1_out_channels, 3, 7)

    layers = [3, 4, 6, 3]
    channels = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)] # (in_channels, mid_channels, out_channels)

    for (in_c, mid_c, out_c) in channels:
        for _ in range(layers[channels.index((in_c, mid_c, out_c))]):
            bottleneck_params = calculate_bottleneck_params(in_c, mid_c, out_c, pruning_ratio=pruning_ratio)
            total_params += bottleneck_params

    return total_params

for ratio in [0.90, 0.92, 0.94, 0.96, 0.98]:
    last_conv_output_channels = 2048
    pruned_last_conv_output_channels = int(last_conv_output_channels * (1 - ratio))
    fc_output_features = 1 # assuming 1 classes for ImageNet

    fc_params = calculate_fc_params(fc_output_features, pruned_last_conv_output_channels)
    resnet50_params = calculate_resnet50_params(pruning_ratio=ratio)
    total_params_with_fc = resnet50_params + fc_params

    print(f"Parameters after {100*ratio}% pruning: {total_params_with_fc / 1e6} M")

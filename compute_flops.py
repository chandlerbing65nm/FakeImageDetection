

def calculate_conv_flops(out_channels, in_channels, kernel_size, output_size, stride=1):

    kernel_ops = kernel_size * kernel_size * in_channels / (stride * stride)
    return 2 * kernel_ops * out_channels * output_size * output_size

def calculate_bottleneck_flops(in_channels, mid_channels, out_channels, output_size, stride=1, pruning_ratio=0.5):

    mid_channels_pruned = int(mid_channels * (1 - pruning_ratio))
    out_channels_pruned = int(out_channels * (1 - pruning_ratio))

    conv1_flops = calculate_conv_flops(mid_channels_pruned, in_channels, 1, output_size, stride)

    conv2_flops = calculate_conv_flops(mid_channels_pruned, mid_channels_pruned, 3, output_size)

    conv3_flops = calculate_conv_flops(out_channels_pruned, mid_channels_pruned, 1, output_size)

    return conv1_flops + conv2_flops + conv3_flops

def calculate_resnet50_flops(input_size, pruning_ratio=0.0):

    conv1_out_channels = 64
    output_size = input_size // 2

    total_flops = calculate_conv_flops(conv1_out_channels, 3, 7, output_size, stride=2)

    layers = [3, 4, 6, 3]
    channels = [(64, 64, 256), (256, 128, 512), (512, 256, 1024), (1024, 512, 2048)]  # (in_channels, mid_channels, out_channels)

    for i, (in_c, mid_c, out_c) in enumerate(channels):
        for _ in range(layers[i]):
            bottleneck_flops = calculate_bottleneck_flops(in_c, mid_c, out_c, output_size, pruning_ratio=pruning_ratio)
            total_flops += bottleneck_flops


            if i < len(layers) - 1:
                output_size = output_size // 2

    return total_flops

def calculate_fc_flops(out_features, in_features):

    return 2 * out_features * in_features


for ratio in [0.90, 0.92, 0.94, 0.96, 0.98]:
    last_conv_output_channels = 2048
    pruned_last_conv_output_channels = int(last_conv_output_channels * (1 - ratio))
    fc_output_features = 1

    fc_flops = calculate_fc_flops(fc_output_features, pruned_last_conv_output_channels)
    resnet50_flops = calculate_resnet50_flops(224, pruning_ratio=ratio)
    total_flops_with_fc = resnet50_flops + fc_flops

    print(f"FLOPS after {100*ratio}% pruning: {total_flops_with_fc / 1e9} G")

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Train.py README</title>
</head>
<body>
    <h1>train.py - A Distributed Training Script</h1>

    <h2>Overview</h2>
    <p>
        The <code>train.py</code> script is used for distributed training of deep learning models. 
        It supports various architectures including ResNet and Vision Transformer (ViT), 
        and is designed for use with PyTorch.
    </p>

    <h2>Arguments</h2>
    <table border="1">
        <tr>
            <th>Argument</th>
            <th>Description</th>
            <th>Default Value</th>
        </tr>
        <tr>
            <td><code>--local_rank</code></td>
            <td>Local rank for distributed training</td>
            <td>0</td>
        </tr>
        <!-- Add more rows for each argument -->
    </table>

    <h2>Mathematical and Algorithmic Insights</h2>
    <p>
        <ul>
            <li>
                <strong>Optimizer:</strong> Uses AdamW, a variant of the Adam optimizer with weight decay fix.
                The learning rate is set to \(0.0001\).
            </li>
            <li>
                <strong>Loss Function:</strong> Binary Cross-Entropy with Logits (BCEWithLogitsLoss) is employed for binary classification.
            </li>
            <li>
                <strong>Early Stopping:</strong> Implemented to prevent overfitting, configurable via the <code>--early_stop</code> flag.
            </li>
            <li>
                <strong>Model Architectures:</strong> Supports ResNet variants and Vision Transformer (ViT) models, both pretrained and non-pretrained.
            </li>
        </ul>
    </p>

    <h2>How to Run</h2>
    <pre>
        python -m torch.distributed.launch --nproc_per_node=2 train.py -- --args.parse
    </pre>

    <h2>Additional Information</h2>
    <p>
        For more advanced configurations, you can modify the <code>train_opt</code> and <code>val_opt</code> dictionaries.
        These control the data augmentation strategies for the training and validation datasets, respectively.
    </p>

</body>
</html>

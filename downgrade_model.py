import torch

def convert_checkpoint_format(new_checkpoint_path, old_checkpoint_path):
    """
    Converts a model checkpoint from the new zip-file format to the old format.

    Parameters:
    - new_checkpoint_path (str): The path to the checkpoint file saved in the new format.
    - old_checkpoint_path (str): The path where the checkpoint will be saved in the old format.
    """
    # Load the checkpoint in the new format
    checkpoint = torch.load(new_checkpoint_path, map_location='cpu')
    
    # Save the checkpoint in the old format
    torch.save(checkpoint, old_checkpoint_path, _use_new_zipfile_serialization=False)

    print(f"Checkpoint was converted and saved to {old_checkpoint_path}")

# Example usage
if __name__ == "__main__":
    # Path to the new format checkpoint
    old_checkpoint = "/home/users/chandler_doloriel/chandler/DeepFakePruning/weights/rn50_modft.pth"
    
    # Path to save the old format checkpoint
    new_checkpoint = "checkpoints/mask_0/rn50_modft.pth"
    
    convert_checkpoint_format(new_checkpoint, old_checkpoint)

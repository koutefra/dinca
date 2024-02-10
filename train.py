import yaml
import sys
from src.train_manager import TrainingManager
from src.loader import DataLoader, ModelSetupManager



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config_path, device):
    config = load_config(config_path)
    config['device'] = device  # Assuming the TrainingManager can handle the device configuration in this way
    training_manager = TrainingManager(config, DataLoader(), ModelSetupManager())
    training_manager.train()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_config_file> [device]")
        sys.exit(1)

    config_file_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"  # Set device to 'cpu' if not specified
    main(config_file_path, device)
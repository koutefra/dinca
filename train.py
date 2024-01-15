import yaml
from src.train_manager import TrainingManager
from src.loader import DataLoader, ModelSetupManager



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config_path):
    config = load_config(config_path)
    training_manager = TrainingManager(config, DataLoader(), ModelSetupManager())
    training_manager.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    main(config_file_path)
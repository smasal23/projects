from pathlib import Path
from src.utils.config import load_yaml


def test_paths_yaml_loads():
    path = Path("configs/paths.yaml")
    data = load_yaml(path)
    assert isinstance(data, dict)
    assert "paths" in data


def test_classification_config_loads():
    path = Path("configs/classification_config.yaml")
    data = load_yaml(path)
    assert "dataset" in data
    assert "expected_classes" in data["dataset"]


def test_classification_config_has_required_keys():
    config_path = Path("configs/classification_config.yaml")
    config = load_yaml(config_path)

    assert "dataset" in config
    assert "expected_splits" in config["dataset"]
    assert "expected_classes" in config["dataset"]
    assert "loader" in config["dataset"]
    assert "normalization" in config["dataset"]
    assert "validation" in config["dataset"]


def test_transfer_learning_config_has_required_training_keys():
    config_path = Path("configs/transfer_learning_config.yaml")
    config = load_yaml(config_path)

    assert "training" in config
    assert "image_size" in config["training"]
    assert "batch_size" in config["training"]
    assert config["training"]["image_size"] == [224, 224]


def test_custom_cnn_config_has_required_sections():
    config_path = Path("configs/classification_config.yaml")
    config = load_yaml(config_path)

    assert "custom_cnn" in config
    assert "training" in config["custom_cnn"]
    assert "augmentation" in config["custom_cnn"]
    assert "model" in config["custom_cnn"]
    assert "callbacks" in config["custom_cnn"]
    assert "artifacts" in config["custom_cnn"]


def test_custom_cnn_training_keys_exist():
    config_path = Path("configs/classification_config.yaml")
    config = load_yaml(config_path)

    training_cfg = config["custom_cnn"]["training"]

    assert "image_size" in training_cfg
    assert "batch_size" in training_cfg
    assert "epochs" in training_cfg
    assert "optimizer" in training_cfg
    assert "learning_rate" in training_cfg
    assert "label_mode" in training_cfg


def test_transfer_learning_config_has_required_top_sections():
    config_path = Path("configs/transfer_learning_config.yaml")
    config = load_yaml(config_path)

    assert "experiment" in config
    assert "training" in config
    assert "models" in config
    assert "reporting" in config


def test_transfer_learning_training_keys_exist():
    config_path = Path("configs/transfer_learning_config.yaml")
    config = load_yaml(config_path)

    training_cfg = config["training"]

    assert "seed" in training_cfg
    assert "image_size" in training_cfg
    assert "batch_size" in training_cfg
    assert "label_mode" in training_cfg
    assert "epochs_frozen" in training_cfg
    assert "epochs_finetune" in training_cfg
    assert "frozen_learning_rate" in training_cfg
    assert "finetune_learning_rate" in training_cfg


def test_transfer_learning_models_have_required_keys():
    config_path = Path("configs/transfer_learning_config.yaml")
    config = load_yaml(config_path)

    for model_name, model_cfg in config["models"].items():
        assert "enabled" in model_cfg
        assert "backbone" in model_cfg
        assert "input_shape" in model_cfg
        assert "weights" in model_cfg
        assert "dense_units" in model_cfg
        assert "dropout_rate" in model_cfg
        assert "callbacks" in model_cfg
        assert "artifacts" in model_cfg

import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from src.predict import predict
from src.train import train


@pytest.mark.slow
def test_train_predict(
    tmp_path: Path, cfg_train: DictConfig, cfg_predict: DictConfig
) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then predicting with
    `predict.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_predict: A DictConfig containing a valid prediction configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_predict.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_predict):
        cfg_predict.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(cfg_predict)
    flat_predictions, _ = predict(cfg_predict)

    # Check the shape of flat_predictions
    assert flat_predictions.shape == (10000, 1)

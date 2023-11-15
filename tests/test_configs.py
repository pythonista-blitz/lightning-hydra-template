import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_predict_config(cfg_predict: DictConfig) -> None:
    """Tests the prediction configuration provided by the `cfg_predict` pytest fixture.

    :param cfg_train: A DictConfig containing a valid prediction configuration.
    """
    assert cfg_predict
    assert cfg_predict.data
    assert cfg_predict.model
    assert cfg_predict.trainer

    HydraConfig().set_config(cfg_predict)

    hydra.utils.instantiate(cfg_predict.data)
    hydra.utils.instantiate(cfg_predict.model)
    hydra.utils.instantiate(cfg_predict.trainer)

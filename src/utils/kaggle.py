import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/utils.py#L17
ON_KAGGLE: bool = "KAGGLE_WORKING_DIR" in os.environ


def generate_submission(
    submit_file_path: str, predict_col: str, result: np.ndarray
) -> pd.DataFrame:
    """Generate Kaggle submission DataFrame.

    :param submit_file_path: Path to the Kaggle submission CSV file.
    :param predict_col: Column name for predictions in the submission file.
    :param result: Numpy array containing predictions.
    :return: DataFrame with predictions for Kaggle submission.
    """
    submit = pd.read_csv(submit_file_path)
    submit[predict_col] = result
    return submit

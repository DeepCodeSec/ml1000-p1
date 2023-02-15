#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import logging
import numpy as np
import pandas as pd
from pycaret.classification import *
#
logger = logging.getLogger(__name__)
#


class WineQualityDataset(object):
    """
    | Column        |   Description         |
    |---------------|-----------------------|
    | `fixed acidity`    | |
    | `volatile acidity` | |
    | `citric acid`      | |
    | `residual sugar`   | |
    | `chlorides`        | |
    | `free sulfur dioxide`  | |
    | `total sulfur dioxide` | | 
    | `density`              | |
    | `pH`                   | |
    | `sulphates`            | |
    | `alcohol`              | |
    | `quality`              | |

    """

    def __init__(self, _file:str, _target_col:str, _training_size=0.8, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file, sep=';')

        # Remove unneeded columns
        #if len(_drop_cols) > 0:
        #    logger.info(f"Removing {len(_drop_cols)} column(s) from the original data set.")
        #    self._df = self._df.drop(_drop_cols)

        #add binary classification label
        self._df["new_quality"] = np.where(self._df["quality"] > 6, "high_quality", "standard")
        self._df = self._df.drop(columns=["quality"])
        self._df = self._df.rename(columns={'new_quality':'quality'})
        self._df.reset_index()

        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        self._clf=setup(data=self._df, target=_target_col)

        # https://pycaret.gitbook.io/docs/get-started/functions/train#compare_models
        self._best_model = compare_models()
        logger.info(self._best_model)

    @property
    def filename(self) -> str:
        return self._datafile

    @property
    def classifier(self):
        return self._clf

    @property
    def dataframe(self):
        return self._df

    @property
    def nb_rows(self) -> int:
        return self.dataframe.shape[0]

    @property
    def nb_cols(self) -> int:
        return self.dataframe.shape[1]

    @property
    def best_model(self):
        return self._best_model

    def save_best_model_to(self, _filename:str) -> None:
        # Save the best model to a file
        save_model(self.best_model, _filename)
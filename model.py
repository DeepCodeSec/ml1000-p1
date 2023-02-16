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
    HQ_THRESHOLD = 7
    LBL_HIGH_QUALITY = 1
    LBL_STD_QUALITY = 0

    def __init__(self, _file:str, _target_col:str, _training_size=0.8, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file, sep=';')

        # Remove unneeded columns
        #if len(_drop_cols) > 0:
        #    logger.info(f"Removing {len(_drop_cols)} column(s) from the original data set.")
        #    self._df = self._df.drop(_drop_cols)

        # Recode quality to a binary label
        self._df["new_quality"] = np.where(self._df["quality"] > WineQualityDataset.HQ_THRESHOLD, 
                                           WineQualityDataset.LBL_HIGH_QUALITY, WineQualityDataset.LBL_STD_QUALITY)
        # Rename columns
        self._df = self._df.drop(columns=["quality"])
        self._df = self._df.rename(columns={'new_quality':'quality'})
        self._df.reset_index()

        # Capping of outliers
        cols = list(self._df.columns)
        tmp = self._df #creating a temporary to avoid accidentally overwriting the original (let's us compare and verify capping)
        data_clean = self._df
        for col in cols[0:-1]:
            upper_limit = self._df[col].mean() + 3*self._df[col].std() #~95th percentile
            lower_limit = self._df[col].mean() - 3*self._df[col].std() #~5th percentile
            logger.info(f"[{col}] 5th and 95th percentiles identified: {upper_limit} - {lower_limit}")
            
            #col_name = self._df.columns[1]
            #self._df = self._df.query(f"{col_name} >= {upper_limit} and {col_name} <= {lower_limit}")
            data_clean[col] = np.where(tmp[col]> upper_limit, upper_limit, #if above 95th, set to upper
                                    np.where(tmp[col]< lower_limit, lower_limit, #if below 5th, set to lower
                                    tmp[col]))

        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        self._clf=setup(
            data=data_clean, 
            target=_target_col,
            data_split_stratify=True,
            transformation=True,
            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold = 0.7,
            fix_imbalance=True)

        # https://pycaret.gitbook.io/docs/get-started/functions/train#compare_models
        #self._best_model = compare_models(sort='Accuracy')
        #self._tuned_model = tune_model(self._best_model)
        #self._final_model = finalize_model(self._best_model)
        lgb = create_model('lightgbm')
        self._best_model = finalize_model(tune_model(lgb, optimize='Prec.'))
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
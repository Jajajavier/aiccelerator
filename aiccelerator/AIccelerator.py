# pylint: disable=E1120
# pylint: disable=E1101
import ppscore as pps
import lazypredict
import numpy as np
import pandas as pd
from uuid import UUID
from typing import List, Optional
from pandas import DataFrame
from lazypredict.Supervised import LazyRegressor
from lazypredict import Supervised
from sklearn.pipeline import Pipeline
from aiccelerator import Utils, HyperRegressionGrid
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


class AIccelerator():
    def __init__(self,
                 *,
                 data: DataFrame = DataFrame(),
                 correlations: Optional[DataFrame] = None,
                 predictive_power_score: Optional[DataFrame] = None,
                 target: Optional[str] = None,
                 predictors: Optional[DataFrame] = None,
                 selected_columns: Optional[List[str]] = None,
                 models: Optional[DataFrame] = None,
                 selected_model_name: Optional[str] = None,
                 param_grid: Optional[dict] = None,
                 preprocessor: Optional[ColumnTransformer] = None,
                 pipeline = None,
                 X_train = None, 
                 X_test= None, 
                 y_train = None, 
                 y_test = None,
                 metadata: List[str] = []) -> None:
        self.data = data
        self.correlations = correlations
        self.predictive_power_score = predictive_power_score
        self.target = target
        self.predictors = predictors
        self.selected_columns = selected_columns
        self.models = models
        self.selected_model_name = selected_model_name
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.pipeline = pipeline
        self.X_train = X_train 
        self.X_test= X_test 
        self.y_train = y_train 
        self.y_test = y_test
        self.metadata = metadata

    def get_correlations(self, method: str = 'pearson', ) -> DataFrame:
        self.correlations = self.data.corr(method=method).dropna(axis=0,how='all').dropna(axis=1,how='all')
        return self.correlations.style.apply(Utils.background_gradients,
                                             cmap='RdBu',
                                             m=-1,
                                             M=1).highlight_null('white')

    def get_predictive_power_score(self) -> DataFrame:
        self.predictive_power_score = pps.matrix(
            self.data)[['x', 'y', 'ppscore']].pivot(index="y",
                                                    columns="x",
                                                    values="ppscore")
        return self.predictive_power_score.style.apply(
            Utils.background_gradients, cmap='Blues', m=0,
            M=1).highlight_null('white')

    def get_predictors(self) -> DataFrame:
        if self.target is not None:
            self.predictors = pps.predictors(
                self.data, y=self.target)[['x', 'ppscore']].set_index(['x'])
        return self.predictors

    def select_columns(self) -> None:
        if self.selected_columns is not None:
            self.data[self.selected_columns]

    def aiccelerate_regressor(self,
                              auto_select_predictors: bool = False,
                              random_split: bool = False,
                              test_size: float = 0.3,
                              relevant_correlations_factor: float = 0.3,
                              relevant_predictors_factor: float = 0.3,
                              ignore_warnings=True) -> None:
        if auto_select_predictors:
            pass
           # correlations = get_correlations()[target]
            #predictors = get_predictors()
        X = self.data[self.selected_columns]
        Y = self.data[self.target]
        if random_split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, Y, test_size=test_size, shuffle=False)

        reg = LazyRegressor(verbose=0, ignore_warnings=ignore_warnings)
        self.models, _ = reg.fit(self.X_train, self.X_test, self.y_train, self.y_test)

    def make_pipeline(self):
        if self.models is not None:
            _, modelpackage = self.getregressorbyname(self.models.iloc[0].name)
            numeric_transformer= Supervised.numeric_transformer
            categorical_transformer = Supervised.categorical_transformer
            if type(self.X_train) is np.ndarray:
                self.X_train = pd.DataFrame(self.X_train)
                self.X_test = pd.DataFrame(self.X_test)

            numeric_features = self.X_train.select_dtypes(
                include=['int64', 'float64', 'int32', 'float32']).columns
            categorical_features = self.X_train.select_dtypes(
                include=['object']).columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric', numeric_transformer, numeric_features),
                    ('categorical', categorical_transformer, categorical_features)
                ])
            
            if 'random_state' in modelpackage().get_params().keys():
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', modelpackage(random_state = 42))
                ])
            else:
                pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', modelpackage())
            ])
            pipe.fit(self.X_train, self.y_train)
            self.pipeline = pipe

    def getregressorbyname(self,modelname:Optional[str]):
        modelname = modelname if modelname else self.models.iloc[0].name
        for name, model in Supervised.REGRESSORS:
            if name==modelname:
                return name,model


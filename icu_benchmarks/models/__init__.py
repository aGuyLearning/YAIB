from typing import Union

from icu_benchmarks.models.dl_models.rnn import GRUNet, LSTMNet, RNNet
from icu_benchmarks.models.dl_models.tcn import TemporalConvNet
from icu_benchmarks.models.dl_models.ts2vec import TS2Vec, TS2VecProbe
from icu_benchmarks.models.dl_models.transformer import BaseTransformer, LocalTransformer, Transformer
from icu_benchmarks.models.ml_models.catboost import CBClassifier
try:
    from icu_benchmarks.models.ml_models.imblearn import BRFClassifier, RUSBClassifier
except ImportError:
    BRFClassifier = None  # type: ignore[assignment,misc]
    RUSBClassifier = None  # type: ignore[assignment,misc]
from icu_benchmarks.models.ml_models.lgbm import LGBMClassifier, LGBMRegressor
from icu_benchmarks.models.ml_models.sklearn import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    MLPClassifier,
    MLPRegressor,
    PerceptronClassifier,
    RFClassifier,
    SVMClassifier,
    SVMRegressor,
)
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier

DLModel = Union[
    GRUNet,
    RNNet,
    LSTMNet,
    TemporalConvNet,
    TS2Vec,
    TS2VecProbe,
    BaseTransformer,
    Transformer,
    LocalTransformer,
]
MLModelClassifier = Union[
    XGBClassifier,
    LGBMClassifier,
    CBClassifier,
    LogisticRegression,
    SVMClassifier,
    PerceptronClassifier,
    MLPClassifier,
    RFClassifier,
]
MLModelRegression = Union[
    MLPRegressor,
    ElasticNet,
    LinearRegression,
    SVMRegressor,
    LGBMRegressor,
]

__all__ = [
    "GRUNet",
    "RNNet",
    "LSTMNet",
    "TemporalConvNet",
    "TS2Vec",
    "TS2VecProbe",
    "BaseTransformer",
    "Transformer",
    "LocalTransformer",
    "CBClassifier",
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBClassifier",
    "LogisticRegression",
    "LinearRegression",
    "ElasticNet",
    "RFClassifier",
    "SVMClassifier",
    "SVMRegressor",
    "MLPRegressor",
    "MLPClassifier",
    "PerceptronClassifier",
    *([n for n in ["RUSBClassifier", "BRFClassifier"] if globals().get(n) is not None]),
]

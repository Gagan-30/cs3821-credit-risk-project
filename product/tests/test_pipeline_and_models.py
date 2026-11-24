import os
import tempfile
import unittest
from product import (
    Instance,
    Dataset,
    CSVParser,
    XLSParser,
    MinMaxScaler,
    ZScoreStandardizer,
    KNNClassifier,
    DecisionTreeClassifier,
)
from product.tests.synthetic_data import (
    generate_linear_clusters,
    generate_quadrant_clusters,
    generate_concentric_rings,
    load_stratified_credit_sample,
)

CREDIT_XLS_PATH = "default of credit card clients.xls"


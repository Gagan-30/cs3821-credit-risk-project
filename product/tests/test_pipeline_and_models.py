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


class TestDataPipeline(unittest.TestCase):
    """Verifies Instance, Dataset, CSVParser, XLSParser, and leakage-free preprocessors."""

    def test_instance_and_dataset_operations(self):
        inst1 = Instance([1, 2.5], label=0)
        inst2 = Instance([3.0, 4], label=1)
        ds = Dataset([inst1, inst2], feature_names=["f0", "f1"])

        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_features, 2)
        self.assertEqual(ds.get_column(1), [2.5, 4.0])
        self.assertEqual(ds.get_labels(), [0, 1])

        train_part, test_part = ds.split(0.5)
        self.assertEqual(len(train_part), 1)
        self.assertEqual(len(test_part), 1)

    
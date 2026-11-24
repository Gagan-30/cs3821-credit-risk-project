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

    def test_csv_parser(self):
        csv_content = "age,income,default\n25,50000,0\n40,20000,1\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        try:
            ds = CSVParser.load(tmp_path)
            self.assertEqual(ds.feature_names, ["age", "income"])
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0].features, [25.0, 50000.0])
            self.assertEqual(ds[1].label, 1)
        finally:
            os.remove(tmp_path)

    @unittest.skipUnless(os.path.exists(CREDIT_XLS_PATH), "Credit card .xls file not present")
    def test_xls_parser_credit_card_dataset(self):
        ds = XLSParser.load_credit_card_default(CREDIT_XLS_PATH, clean_categories=True, rename_pay0=True)
        self.assertEqual(len(ds), 30000)
        self.assertEqual(ds.num_features, 23)
        self.assertEqual(ds.feature_names[0], "LIMIT_BAL")
        self.assertEqual(ds.feature_names[5], "PAY_1")
        self.assertEqual(ds[0].meta["id"], 1)

        edu_vals = set(ds.get_column(ds.feature_names.index("EDUCATION")))
        mar_vals = set(ds.get_column(ds.feature_names.index("MARRIAGE")))
        self.assertTrue(edu_vals.issubset({1.0, 2.0, 3.0, 4.0}))
        self.assertTrue(mar_vals.issubset({1.0, 2.0, 3.0}))

        subset = Dataset(ds.instances[:50], feature_names=ds.feature_names)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            XLSParser.export_to_csv(subset, tmp_path)
            reloaded = CSVParser.load(tmp_path)
            self.assertEqual(len(reloaded), 50)
            self.assertEqual(reloaded.num_features, 23)
            self.assertEqual(reloaded[0].features, subset[0].features)
            self.assertEqual(reloaded[0].label, subset[0].label)
        finally:
            os.remove(tmp_path)

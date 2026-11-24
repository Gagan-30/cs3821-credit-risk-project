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

    def test_minmax_scaler_no_leakage(self):
        train = Dataset([Instance([10.0, 100.0], 0), Instance([20.0, 200.0], 1)])
        test = Dataset([Instance([30.0, 150.0], 1)])

        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)

        self.assertEqual(scaled_train[0].features, [0.0, 0.0])
        self.assertEqual(scaled_train[1].features, [1.0, 1.0])
        self.assertEqual(scaled_test[0].features, [2.0, 0.5])

    def test_zscore_standardizer(self):
        train = Dataset([Instance([10.0], 0), Instance([20.0], 1), Instance([30.0], 0)])
        stdizer = ZScoreStandardizer()
        scaled = stdizer.fit_transform(train)

        self.assertAlmostEqual(scaled[1][0], 0.0, places=6)
        self.assertAlmostEqual(scaled[0][0], -scaled[2][0], places=6)

class TestKNNClassifier(unittest.TestCase):
    """Verifies k-NN distance metrics, synthetic accuracy, tie-breaking rules, and credit card execution."""

    def test_linear_and_concentric_accuracy(self):
        for ds in (generate_linear_clusters(100, seed=7), generate_concentric_rings(100, seed=7)):
            train, test = ds.split(0.8)
            for metric in ("euclidean", "manhattan"):
                knn = KNNClassifier(k=3, metric=metric)
                knn.fit(train)
                preds = knn.predict(test)
                acc = sum(p == y for p, y in zip(preds, test.get_labels())) / len(test)
                self.assertGreaterEqual(acc, 0.95)

    def test_tie_breaking_strategies(self):
        train = Dataset([
            Instance([0.0], label=0),
            Instance([0.1], label=0),
            Instance([1.0], label=0),
            Instance([1.8], label=1),
            Instance([5.0], label=1),
        ])
        query = Instance([1.45])

        knn_weighted = KNNClassifier(k=2, tie_breaker="distance_weighted")
        knn_weighted.fit(train)
        self.assertEqual(knn_weighted.predict_instance(query), 1)

        knn_reduction = KNNClassifier(k=2, tie_breaker="neighbor_reduction")
        knn_reduction.fit(train)
        self.assertEqual(knn_reduction.predict_instance(query), 1)

        knn_prior = KNNClassifier(k=2, tie_breaker="class_prior")
        knn_prior.fit(train)
        self.assertEqual(knn_prior.predict_instance(query), 0)

    @unittest.skipUnless(os.path.exists(CREDIT_XLS_PATH), "Credit card .xls file not present")
    def test_credit_card_toy_pipeline_knn(self):
        ds = load_stratified_credit_sample(CREDIT_XLS_PATH, n_samples=150, seed=42)
        train_raw, test_raw = ds.split(0.8)

        scaler = ZScoreStandardizer()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        for tb in ("majority", "distance_weighted", "neighbor_reduction", "class_prior"):
            knn = KNNClassifier(k=5, metric="euclidean", tie_breaker=tb)
            knn.fit(train_scaled)
            preds = knn.predict(test_scaled)
            probas = knn.predict_proba(test_scaled)
            self.assertEqual(len(preds), len(test_scaled))
            self.assertTrue(all(p in (0, 1) for p in preds))
            self.assertTrue(all(abs(sum(pr) - 1.0) < 1e-6 for pr in probas))


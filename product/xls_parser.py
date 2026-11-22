import csv
from typing import Optional, List, Any
from .dataset import Dataset
from .instance import Instance


class XLSParser:
    """
    Parses .xls Excel files into native Dataset objects without external ML dependencies.
    Uses xlrd directly (with pandas fallback if available) and supports optional CSV caching.
    """

    @staticmethod
    def _read_sheet_rows(filepath: str, sheet_name: Optional[str] = None) -> List[List[Any]]:
        try:
            import xlrd
            workbook = xlrd.open_workbook(filepath)
            sheet = workbook.sheet_by_name(sheet_name) if sheet_name else workbook.sheet_by_index(0)
            return [sheet.row_values(r) for r in range(sheet.nrows)]
        except ImportError:
            import pandas as pd
            target_sheet = sheet_name if sheet_name is not None else 0
            df = pd.read_excel(filepath, sheet_name=target_sheet, header=None)
            return df.values.tolist()

    @staticmethod
    def load_credit_card_default(
        filepath: str,
        sheet_name: str = "Data",
        clean_categories: bool = True,
        rename_pay0: bool = True,
        limit: Optional[int] = None,
    ) -> Dataset:
        """
        Loads the UCI Default of Credit Card Clients (.xls) dataset.

        Structure of 'default of credit card clients.xls':
          - Row 0: Variable codes (X1..X23, Y)
          - Row 1: Column names ('ID', 'LIMIT_BAL', ..., 'default payment next month')
          - Rows 2..30001: 30,000 client instances (ID + 23 features + binary default target)
        """
        rows = XLSParser._read_sheet_rows(filepath, sheet_name=sheet_name)
        if len(rows) < 3:
            raise ValueError(f"Expected at least 3 rows in '{filepath}', found {len(rows)}.")

        headers = [str(h).strip() for h in rows[1]]
        feature_names = headers[1:-1]
        if rename_pay0 and "PAY_0" in feature_names:
            idx = feature_names.index("PAY_0")
            feature_names[idx] = "PAY_1"

        edu_idx = feature_names.index("EDUCATION") if "EDUCATION" in feature_names else -1
        mar_idx = feature_names.index("MARRIAGE") if "MARRIAGE" in feature_names else -1

        dataset = Dataset(feature_names=feature_names)
        data_rows = rows[2:] if limit is None else rows[2 : 2 + limit]

        for row in data_rows:
            if not row or row[0] == "" or row[0] is None:
                continue

            record_id = int(float(row[0]))
            features = [float(val) for val in row[1:-1]]
            label = int(float(row[-1]))

            if clean_categories:
                # Documented EDUCATION: 1=grad school, 2=university, 3=high school, 4=others
                if edu_idx != -1 and features[edu_idx] in (0.0, 5.0, 6.0):
                    features[edu_idx] = 4.0
                # Documented MARRIAGE: 1=married, 2=single, 3=others
                if mar_idx != -1 and features[mar_idx] == 0.0:
                    features[mar_idx] = 3.0

            dataset.add(Instance(features=features, label=label, meta={"id": record_id}))

        return dataset

    @staticmethod
    def export_to_csv(dataset: Dataset, output_csv_path: str, target_name: str = "default") -> None:
        """Exports a loaded Dataset to a standard flat CSV file compatible with CSVParser."""
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if dataset.feature_names:
                writer.writerow(list(dataset.feature_names) + [target_name])
            for inst in dataset:
                writer.writerow(list(inst.features) + [inst.label])
import csv
from typing import Optional, List, Any
from .dataset import Dataset
from .instance import Instance


class CSVParser:
    """Parses flat CSV files into Dataset objects."""

    @staticmethod
    def load(
        filepath: str,
        target_col: int = -1,
        has_header: bool = True,
        categorical_map: Optional[dict] = None
    ) -> Dataset:
        dataset = Dataset()
        categorical_map = categorical_map or {}

        with open(filepath, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            
            if has_header:
                headers = next(reader)
                if target_col == -1:
                    dataset.feature_names = headers[:-1]
                else:
                    dataset.feature_names = [h for i, h in enumerate(headers) if i != target_col]

            for row in reader:
                if not row or all(val.strip() == "" for val in row):
                    continue

                raw_label = row[target_col] if target_col != -1 else row[-1]
                raw_features = row[:target_col] if target_col == -1 else [val for i, val in enumerate(row) if i != target_col]

                # Convert features to float
                parsed_features: List[float] = []
                for val in raw_features:
                    val_str = val.strip()
                    if val_str in categorical_map:
                        parsed_features.append(float(categorical_map[val_str]))
                    else:
                        parsed_features.append(float(val_str))

                # Parse label
                label_str = raw_label.strip()
                parsed_label: Any = categorical_map.get(label_str, label_str)
                try:
                    parsed_label = int(parsed_label)
                except ValueError:
                    pass

                dataset.add(Instance(features=parsed_features, label=parsed_label))

        return dataset
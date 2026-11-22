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

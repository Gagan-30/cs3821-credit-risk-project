from pathlib import Path
from typing import Optional
from ..dataset import Dataset
from ..xls_parser import XLSParser

CREDIT_CARD_XLS_PATH: str = str(
    Path(__file__).resolve().parent / "default of credit card clients.xls"
)


def load_credit_card_default(
    filepath: str = CREDIT_CARD_XLS_PATH,
    sheet_name: str = "Data",
    clean_categories: bool = True,
    rename_pay0: bool = True,
    limit: Optional[int] = None,
) -> Dataset:
    """Convenience loader for the bundled UCI Default of Credit Card Clients dataset."""
    return XLSParser.load_credit_card_default(
        filepath=filepath,
        sheet_name=sheet_name,
        clean_categories=clean_categories,
        rename_pay0=rename_pay0,
        limit=limit,
    )


__all__ = [
    "CREDIT_CARD_XLS_PATH",
    "load_credit_card_default",
]
"""Data platform contracts and Pandera schemas for items and interactions."""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when data contracts or schemas fail validation."""

    pass


def _format_schema_errors(dataset_name: str, exc: pa.errors.SchemaErrors) -> str:
    """Format Pandera SchemaErrors into clear, readable error messages."""
    failure_cases = exc.failure_cases
    messages: list[str] = []

    for _, row in failure_cases.iterrows():
        check_name = str(row.get("check", ""))
        col_name = row.get("column")
        failure_case = row.get("failure_case")

        if check_name == "column_in_dataframe" or col_name is None:
            messages.append(f"Required column '{failure_case}' is missing from DataFrame.")
        elif check_name.startswith("greater_than_or_equal_to"):
            messages.append(
                f"Column '{col_name}' has values not greater than or equal to 0 "
                f"(found '{failure_case}')."
            )
        elif check_name.startswith("str_length"):
            messages.append(
                f"Column '{col_name}' has invalid empty or short string (found '{failure_case}')."
            )
        else:
            messages.append(
                f"Column '{col_name}' failed validation check '{check_name}' "
                f"with value '{failure_case}'."
            )

    joined = "\n  - ".join(messages)
    return f"{dataset_name} validation failed with {len(messages)} issue(s):\n  - {joined}"


ItemsSchema = DataFrameSchema(
    columns={
        "item_id": Column(int, Check.ge(0), nullable=False, coerce=True),
        "title": Column(str, Check.str_length(min_value=1), nullable=False),
        "genres": Column(str, required=False, nullable=True),
        "description": Column(str, required=False, nullable=True),
        "attributes": Column(object, required=False, nullable=True),
    },
    strict=False,
    coerce=True,
)

InteractionsSchema = DataFrameSchema(
    columns={
        "user_id": Column(int, Check.ge(0), nullable=False, coerce=True),
        "item_id": Column(int, Check.ge(0), nullable=False, coerce=True),
        "event_type": Column(str, Check.str_length(min_value=1), nullable=False),
        "timestamp": Column(int, Check.ge(0), nullable=False, coerce=True),
        "value": Column(float, required=False, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)


def validate_items(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the items contract.

    Required columns:
    - item_id: int >= 0
    - title: non-empty string

    Optional columns:
    - genres: string
    - description: string
    - attributes: object / JSON string / dict
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected pandas DataFrame for items, got {type(df).__name__}")
    try:
        return ItemsSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        msg = _format_schema_errors("Items", exc)
        raise DataValidationError(msg) from exc
    except Exception as exc:
        raise DataValidationError(f"Items validation failed: {exc}") from exc


def validate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the interactions contract.

    Required columns:
    - user_id: int >= 0
    - item_id: int >= 0
    - event_type: non-empty string (e.g. 'rating', 'click', 'purchase', 'view')
    - timestamp: int >= 0

    Optional columns:
    - value: float / numeric rating or intensity
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"Expected pandas DataFrame for interactions, got {type(df).__name__}"
        )
    try:
        return InteractionsSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        msg = _format_schema_errors("Interactions", exc)
        raise DataValidationError(msg) from exc
    except Exception as exc:
        raise DataValidationError(f"Interactions validation failed: {exc}") from exc

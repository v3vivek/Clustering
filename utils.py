import copy
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import unidecode

from common.tai_g_um_utils.utils import safe_divide


def merge_dicts(x) -> dict:
    """Merges dictionaries

    Args:
        x (list): List of dictionaries

    Returns:
        dict: merged dict
    """
    x = [k for k in x if isinstance(k, dict)]
    return {k: v for d in x for k, v in d.items()}


def dict_without_some_keywords(in_dict, some_keys: Optional[list] = None):
    if some_keys is None:
        some_keys = []

    if isinstance(in_dict, dict) is False:
        return in_dict
    out_dict = copy.deepcopy(in_dict)

    for key in some_keys:
        out_dict.pop(key, None)
    return out_dict


def is_null(x, additional_nulls: Optional[list] = None, data_type=None) -> bool:
    """Check is value is none

    Args:
        x (Any): Value to check for null
        additional_nulls (list, optional): Additional value to consider as null. Defaults to [].
        data_type (dtype, optional): Data type to check for null. Defaults to str.

    Returns:
        bool: True if input is null else False
    """
    if additional_nulls is None:
        additional_nulls = []
    none_vals = [None, np.nan, np.inf]
    none_vals.extend(additional_nulls)
    if x in none_vals:
        return True

    try:
        if data_type and isinstance(x, data_type):
            return False
        if isinstance(x, str) and x.lower() not in ["nan", "inf", "none", "null"]:
            return False
        if isinstance(x, str) and x.lower() in ["nan", "inf", "none", "null"]:
            return True
        if isinstance(x, (int, float, np.number)) and (np.isnan(x) or pd.isna(x)):  # type: ignore
            return True
    except Exception:
        pass

    return False


def isnumeric(x, dtype=float) -> bool:
    """Check if the passed value isnumeric

    Args:
        x (Any): Passed value
        dtype (Any, optional): Datatype check. Defaults to float.

    Returns:
        bool: True if isnumeri is true else false
    """
    try:
        dtype(str(x))
        return True
    except Exception:
        return False


def scale_between(
    unscaled_num: float, min_allowed: float, max_allowed: float, min_out: float, max_out: float, round_upto: int = 3
) -> float | None:
    """Scale the number using min max scoring

    Args:
        unscaled_num (float): score to be scaled
        min_allowed (float): min input score
        max_allowed (float): max input score
        min_out (float): min output score
        max_out (float): max output score
        round_upto (int): round digits

    Returns:
        float: scaled score
    """
    try:
        return float(
            np.round(
                (max_allowed - min_allowed) * (unscaled_num - min_out) / (max_out - min_out) + min_allowed,
                round_upto,
            )
        )
    except Exception:
        return None


def compute_sqrt_mean(x: list, error_val: Any = None, skipna: bool = False) -> float:
    """Compute square root mean

    Args:
        x (Any): List of numbers
        error_val (Any, optional): Error val. Defaults to None.
        skipna (bool, optional): Indicator to skip nulls. Defaults to False.

    Returns:
        float: Square root mean value
    """
    try:
        if skipna is True:
            x = [c for c in x if is_null(c) is False]
        val = np.sqrt(np.mean(np.square(x)))
        if is_null(val) is False:
            return val
        else:
            return error_val
    except Exception:
        return error_val


def compute_mean(x: list, error_val: Any = None, skipna: bool = False) -> float:
    """Compute mean
    Args:
        x (Any): List of numbers
        error_val (Any, optional): Error val. Defaults to None.
        skipna (bool, optional): Indicator to skip nulls. Defaults to False.
    Returns:
        float: mean value
    """
    try:
        if skipna is True:
            x = [c for c in x if is_null(c) is False]
        val = np.mean(x)
        if is_null(val) is False:
            return float(val)
        else:
            return error_val
    except Exception:
        return error_val


def chunks(list_to_chunk, chunk_size):
    """Returns a list with chunks of chunk_size from the list_to_chunk

    Args:
        list_to_chunk (list): List to be chunked
        chunk_size (int): Chunk size

    Returns:
        list: List of chunk_size
    """
    # return chunked array
    output = []
    for iter_id in range(0, len(list_to_chunk), chunk_size):
        output.append(list_to_chunk[iter_id : iter_id + chunk_size])
    return output


def calculate_additional_cost_metrics(
    df: pd.DataFrame,
    scale_metric: str | None = None,
    suffixes: str = "",
    in_micros: bool = False,
) -> pd.DataFrame:
    """Calculate additional cost-related metrics based on existing columns in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the relevant data.
        scale_metric (str, optional): Scale metric for calculating additional metrics. Defaults to None.
        suffixes (str, optional): Suffixes for columns, e.g., "_adg", "_tag". Defaults to "".
        in_micros (bool, optional): Indicates if cost metrics are in micros. Defaults to False.
            If False, cost metrics will be divided by 1,000,000.

    Returns:
        pd.DataFrame: DataFrame with additional cost-related metrics calculated.
    """
    # Determine the divisor for cost metrics based on in_micros flag
    df = df.copy()
    micros_divider = 1 if in_micros else 1_000_000

    # Adjust suffixes format
    suffixes = f"_{suffixes}" if suffixes and not suffixes.startswith("_") else suffixes

    # Reset index for consistency
    df.reset_index(drop=True, inplace=True)

    # Calculate cost-related metrics if the cost column exists in DataFrame
    cost_micros_col = f"metrics.costMicros{suffixes}"
    if cost_micros_col in df:
        df[cost_micros_col] = pd.to_numeric(df[cost_micros_col])
        # this will always be divided by micros_divider irrespective of in_micros flag
        df[f"metrics.cost{suffixes}"] = (
            df[f"metrics.costMicros{suffixes}"] / micros_divider
        )
        # calculate cpr if scale metric column exists
        if f"{scale_metric}{suffixes}" in df:
            df[f"cpr{suffixes}"] = safe_divide(df[f"metrics.cost{suffixes}"], df[f"{scale_metric}{suffixes}"])

        # Calculate CPC if clicks column exists
        clicks_col = f"metrics.clicks{suffixes}"
        if clicks_col in df:
            cpc_col = f"cpc{suffixes}"
            df[cpc_col] = np.inf
            df[clicks_col] = pd.to_numeric(df[clicks_col])

            # Avoid division by zero and calculate CPC
            df.loc[df[clicks_col] > 0, cpc_col] = (df[cost_micros_col] / micros_divider) / df[clicks_col]

        else:
            print(f"{clicks_col} not in df")

        # Calculate CPA if conversions column exists
        conv_col = f"metrics.conversions{suffixes}"
        if conv_col in df:
            cpa_col = f"cpa{suffixes}"
            df[cpa_col] = np.inf
            df[conv_col] = pd.to_numeric(df[conv_col])

            # Avoid division by zero and calculate CPA
            df.loc[df[conv_col] > 0, cpa_col] = (df[cost_micros_col] / micros_divider) / df[conv_col]

        else:
            print(f"{conv_col} not in df")

        # Calculate ROAS if conversion value column exists
        conv_val_col = f"metrics.conversionsValue{suffixes}"
        if conv_val_col in df:
            roas_col = f"roas{suffixes}"
            df[roas_col] = 0.0
            df[conv_val_col] = pd.to_numeric(df[conv_val_col]).astype(float)

            # Avoid division by zero and calculate ROAS
            df.loc[df[conv_val_col] > 0, roas_col] = df[conv_val_col] / (df[cost_micros_col] / micros_divider)

        else:
            print(f"{conv_val_col} not in df")

    else:
        print(f"{cost_micros_col} not in df")

    return df


def normalize_series(
    _series: pd.Series,
    old_max=None,
    old_min=None,
    new_min: int = 0,
    new_max: int = 100,
    reverse_score: bool = False,
    verbose: bool = True,
) -> pd.Series:
    """Normalize numeric type series; otherwise, return the same series.

    Args:
        _series (pd.Series): The input series.
        old_max (int, optional): Old max -> new_max. In case of none, series max is old_max. Defaults to None.
        old_min (int, optional): Old min -> new_min. In case of none, series min is old_min. Defaults to None.
        new_min (int, optional): Min value of the new series. Defaults to 0.
        new_max (int, optional): Max value of the new series. Defaults to 100.
        reverse_score (bool, optional): Assign new_max to new_min and new_min to new_max, and normalize the series.
                                        Defaults to False.

    Returns:
        pd.Series: Normalized series, or the same for non-numeric series
    """

    try:
        if reverse_score:
            new_min, new_max = new_max, new_min

        if ".id" not in str(_series.name):
            _series = pd.to_numeric(_series, errors="raise")
            if (not old_max) or (not old_min):
                old_min, old_max = _series.min(), _series.max()

            # Clip values less than old_min to old_min and values greater than old_max to old_max
            _series = _series.clip(lower=old_min, upper=old_max)
            _series = (_series - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
            return _series
        # if skipping and verbose then print
        elif verbose:
            print(f"Skipped as series contains `.id`: {_series.name}")
    except ValueError:
        print(f"{_series.name}: {_series.dtype}")
    return _series


def get_min(values: list, default=None, min_val_to_consider=None):
    valid_values = [val for val in values if (is_null(val) is False)]
    if is_null(min_val_to_consider) is False:
        valid_values = [val for val in valid_values if val > min_val_to_consider]
    if len(valid_values) == 0:
        return default
    return np.min(valid_values)


def get_max(values: list, default=None, max_val_to_consider=None):
    valid_values = [val for val in values if (is_null(val) is False)]
    if is_null(max_val_to_consider) is False:
        valid_values = [val for val in valid_values if val < max_val_to_consider]
    if len(valid_values) == 0:
        return default
    return np.max(valid_values)


def multiply_two_numbers(a, b, error_val=None, multiplier=1, round_digits=None) -> float | None | Any:
    """Multiply two numbers
    # rename this to multiply_two_numbers to avoid confusion with the multiply function
    Args:
        a (Any): First number
        b (Any): Second number
        error_val (Any, optional): Throw in case of error. Defaults to None.
        multiplier (float, optional): Multiplies the outout by this number. Defaults to 1.
        round_digits (int, optional): Output round off digits. Defaults to None.

    Returns:
        float: Multiplication output
    """
    try:
        val = multiplier * np.double(a) * np.double(b)
        if round_digits is not None:
            val = float(np.round(val, round_digits))
        return float(val)
    except Exception:
        return error_val


def division(numerator, denominator, error_val=None, multiplier=1, round_digits=None) -> float | None | Any:
    """Divide two numbers

    Args:
        numerator (Any): Numerator (Parseable to float)
        denominator (Any): Denominator (Parseable to float)
        error_val (Any, optional): Throw in case of error. Defaults to None.
        multiplier (float, optional): Multiplies the outout by this number. Defaults to 1.
        round_digits (int, optional): Output round off digits. Defaults to None.

    Returns:
        float: Division output
    """
    try:
        val = multiplier * np.double(numerator) / np.double(denominator)
        if round_digits is not None:
            val = float(np.round(val, round_digits))
        return float(val)
    except Exception:
        return error_val


def clean_keywords(keyword: str) -> str:
    """Clean keywords
    Args:
        keyword (str): Raw keywords
    Returns:
        str: Cleaned keywords
    """
    keyword = keyword.lower()
    keyword = re.sub(" +", " ", keyword)
    keyword = re.sub(r'["+\[\]]', "", keyword)
    keyword = unidecode.unidecode(keyword)
    return keyword.strip()


def group_sim_from_matrix(matrix, l2_agg=False):

    if l2_agg:
        return (np.sqrt(np.square(matrix.max(axis=0)).mean()) + np.sqrt(np.square(matrix.max(axis=1)).mean())) / 2

    else:
        return (matrix.max(axis=0).mean() + matrix.max(axis=1).mean()) / 2


def group_sim_from_matrix_percentile(matrix, top_n_perc=1.0, l2_agg=False):
    row_elements = matrix.max(axis=0)
    col_elements = matrix.max(axis=1)

    row_elem_count = min(int(np.ceil(row_elements.shape[0] * top_n_perc)), len(row_elements))
    col_elem_count = min(int(np.ceil(col_elements.shape[0] * top_n_perc)), len(col_elements))

    top_row_elements = row_elements[np.argsort(row_elements)][-int(row_elem_count) :]
    top_col_elements = col_elements[np.argsort(col_elements)][-int(col_elem_count) :]

    if l2_agg:
        return (np.sqrt(np.square(top_row_elements).mean()) + np.sqrt(np.square(top_col_elements).mean())) / 2
    else:
        return (top_row_elements.mean() + top_col_elements.mean()) / 2

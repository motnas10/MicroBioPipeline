import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.colors import Normalize

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from sklearn.manifold import TSNE
from collections import namedtuple
from emperor import Emperor
from typing import Callable, Tuple, Optional
import torch
import pandas as pd
from scipy.stats import t
import os

# --------------------------------------------------------------------------------------------------------------
# Define p-value to symbol conversion
def pval_to_symbol(p):
    """
    Convert p-value to a symbol for significance.
    Args:
        p (float): The p-value to convert.
    Returns:
        str: The corresponding symbol.
    """
    if p <= 1e-4:
        return '****'
    elif p <= 1e-3:
        return '***'
    elif p <= 1e-2:
        return '**'
    elif p <= 5e-2:
        return '*'
    else:
        return 'ns'
    
# --------------------------------------------------------------------------------------------------------------
# DataFrame filtering by multiple features
import numpy as np
import pandas as pd

def filter_data(df: pd.DataFrame, filters) -> pd.DataFrame:
    """
    Filter DataFrame by multiple features.

    Semantics:
    - If filters is None or empty -> return a copy of df (no filtering).
    - For each feature:
      * If value is None -> skip filtering that feature.
      * If value is an iterable containing None -> skip filtering that feature.
      * If value is an iterable (list/tuple/set/ndarray/Series) -> apply isin() on non-NA entries.
      * If value is a scalar:
          - if pd.isna(value) -> match df[col].isna()
          - otherwise -> match df[col] == value
    """
    if not filters:
        return df.copy()

    mask = np.ones(len(df), dtype=bool)

    for col, val in filters.items():
        # skip this column entirely if None provided
        if val is None:
            continue

        # handle iterable filters
        if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
            # if the user explicitly included None in the iterable -> skip this column
            if any(v is None for v in val):
                continue

            # remove NA-like entries (np.nan, pd.NA) from the list before isin
            vals = [v for v in val if not pd.isna(v)]
            if not vals:
                # nothing meaningful to filter on
                continue

            mask &= df[col].isin(vals).to_numpy()

        else:
            # scalar filter
            if pd.isna(val):
                mask &= df[col].isna().to_numpy()
            else:
                mask &= (df[col] == val).to_numpy()

        # fast short-circuit: if mask is all False, no rows will match
        if not mask.any():
            return df.iloc[0:0]  # empty DataFrame with same columns

    return df.loc[mask].copy()
# --------------------------------------------------------------------------------------------------------------
# General data loader
def load_data(file_path: str, filetype: str = "excel", **kwargs):
    """
    General loader for excel (single/multi-sheet), csv, txt, and tsv.
    Returns dict of {sheet_or_file_name: DataFrame}

    Parameters:
    - file_path: path to the file
    - filetype: one of "excel", "csv", "txt", "tsv"
    - kwargs: forwarded to pandas read_* functions (e.g., encoding, dtype, etc.)

    Notes:
    - For excel, all sheets are loaded and returned as a dict mapping sheet name -> DataFrame.
    - For csv/txt/tsv, a single-entry dict is returned with key equal to the file type
      (you can change this behavior to use the filename if desired).
    """
    filetype = (filetype or "").lower()

    if filetype == "excel":
        # read all sheets; by default use the first column as index and first row as header
        dfs = pd.read_excel(file_path, sheet_name=None, index_col=0, header=0, **kwargs)
        return dfs
    elif filetype == "csv":
        df = pd.read_csv(file_path, index_col=0, header=0, **kwargs)
        return {"csv": df}
    elif filetype == "txt":
        delimiter = kwargs.pop("delimiter", "\t")
        df = pd.read_csv(file_path, delimiter=delimiter, index_col=0, header=0, **kwargs)
        return {"txt": df}
    elif filetype == "tsv":
        # tab-separated values; allow override via kwargs['delimiter'] if provided
        delimiter = kwargs.pop("delimiter", "\t")
        df = pd.read_csv(file_path, delimiter=delimiter, index_col=0, header=0, **kwargs)
        return {"tsv": df}
    else:
        # Try to infer from extension as a convenience
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip(".")
        if ext in {"xls", "xlsx", "xlsm", "xlsb"}:
            return load_data(file_path, "excel", **kwargs)
        if ext == "csv":
            return load_data(file_path, "csv", **kwargs)
        if ext in {"txt", "tsv"}:
            inferred = "tsv" if ext == "tsv" else "txt"
            return load_data(file_path, inferred, **kwargs)
        raise ValueError(f"Unsupported file type: {filetype}")

# --------------------------------------------------------------------------------------------------------------
# Write DataFrame to Excel sheet
def write_df_to_excel_sheet(df, filename, sheet_name):
    """
    Writes a pandas DataFrame to a new sheet in an Excel file.
    If the file exists, adds the sheet; otherwise, creates a new file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to write.
        filename (str): The name of the Excel file (e.g., 'output.xlsx').
        sheet_name (str): The name of the sheet to write to.
    """
    from pathlib import Path
    if Path(filename).exists():
        # If file exists, append new sheet without overwriting
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        # Create new file with the sheet
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name)

# --------------------------------------------------------------------------------------------------------------
# Convert dict to filename-friendly string
import json
import re
import hashlib
from typing import Any, Dict

def dict_to_filename(d: Dict[str, Any],
                     sort_keys: bool = True,
                     item_sep: str = "_",
                     max_len: int = 120) -> str:
    """
    Convert a dict's values into a filename-friendly string.
    - sorts keys for deterministic output (by default)
    - flattens lists/tuples
    - JSON-encodes nested dicts
    - replaces unsafe filename chars with '_'
    - if resulting string is longer than max_len, returns an MD5 hash suffix
    """
    import json, hashlib, re
    parts = []
    iterator = sorted(d.items()) if sort_keys else d.items()
    for _, v in iterator:
        if isinstance(v, (list, tuple)):
            v_str = "-".join(map(str, v))
        elif isinstance(v, dict):
            v_str = json.dumps(v, sort_keys=sort_keys, separators=(",", ":"))
        else:
            v_str = str(v)
        parts.append(v_str)
    combined = item_sep.join(parts)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", combined)
    if len(safe) <= max_len:
        return safe
    prefix = safe[:max_len - 33].rstrip("_")
    digest = hashlib.md5(safe.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"
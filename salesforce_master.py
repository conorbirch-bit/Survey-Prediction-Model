from __future__ import annotations

import io
import re
from typing import Optional, Tuple

import pandas as pd


# Broad UK postcode matcher. It intentionally accepts the normal postcode forms
# used in the Salesforce Building Name / address field.
UK_POSTCODE_RE = re.compile(
    r"\b("
    r"GIR\s?0AA|"
    r"(?:[A-PR-UWYZ][0-9][0-9]?|"
    r"[A-PR-UWYZ][A-HK-Y][0-9][0-9]?|"
    r"[A-PR-UWYZ][0-9][A-HJKSTUW]|"
    r"[A-PR-UWYZ][A-HK-Y][0-9][ABEHMNPRV-Y])"
    r"\s?[0-9][ABD-HJLNP-UW-Z]{2}"
    r")\b",
    re.IGNORECASE,
)


def _cell_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalise_salesforce_status(value) -> str:
    """
    Normalise Salesforce status spelling variants without changing the
    scheduling rules.

    The current report uses "Work Request". Some exports/users may call the same
    state "Work Requested", so both are standardised to "Work Request".
    """
    text = _cell_text(value)
    if not text:
        return ""

    key = re.sub(r"\s+", " ", text).strip().lower()
    aliases = {
        "work request": "Work Request",
        "work requested": "Work Request",
    }
    return aliases.get(key, text)


def find_header_row(raw: pd.DataFrame) -> int:
    """
    Detect either a normal row-1 header or a Salesforce report header that
    appears after report/filter metadata.
    """
    best_row = None
    best_score = -1

    expected_markers = {
        "customer reference code",
        "building name",
        "building height",
        "sovereign flat",
        "work type name",
        "status",
    }

    for idx, row in raw.iterrows():
        normalised = set()
        for value in row.tolist():
            text = _cell_text(value).lower()
            # Strip Salesforce arrows and repeated spacing.
            text = text.replace("↑", "").strip()
            if text:
                normalised.add(text)

        score = sum(1 for marker in expected_markers if marker in normalised)
        if score > best_score:
            best_score = score
            best_row = int(idx)

    # Require enough identifying fields that a title/filter row cannot win.
    if best_row is None or best_score < 2:
        return 0
    return best_row


def extract_postcode(value) -> str:
    text = _cell_text(value).upper()
    if not text:
        return ""

    matches = list(UK_POSTCODE_RE.finditer(text))
    if not matches:
        return ""

    # Addresses normally end with the postcode. Taking the last match is safer
    # if another postcode-like token appears earlier in descriptive text.
    postcode = matches[-1].group(1).upper()
    postcode = re.sub(r"\s+", "", postcode)

    if len(postcode) > 3:
        return f"{postcode[:-3]} {postcode[-3:]}"
    return postcode


def _forward_fill_grouped_salesforce_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Salesforce grouped reports commonly print Work Type Name / Status only on
    the first row of a group and leave the rows below blank. Carry those two
    classifications down so every building has its own explicit status.
    """
    result = df.copy()

    for col in ["Work Type Name", "Status"]:
        if col in result.columns:
            result[col] = result[col].replace(r"^\s*$", pd.NA, regex=True).ffill()

    # Explicitly support both "Work Request" and "Work Requested".
    if "Status" in result.columns:
        result["Status"] = result["Status"].apply(normalise_salesforce_status)

    return result


def derive_drawing_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business rule supplied for the master list:
      Work Type Name = Plan Drafting
      AND Status = Work Request
      => Needs Drawing

    Geospatial Asset Mapping rows are treated as Ready.

    An explicit Drawing Status column is respected where it is already populated.
    """
    result = df.copy()

    if "Drawing Status" not in result.columns:
        result["Drawing Status"] = pd.NA

    existing = result["Drawing Status"].astype("string")
    work_type = (
        result["Work Type Name"].astype("string").str.strip().str.lower()
        if "Work Type Name" in result.columns
        else pd.Series("", index=result.index, dtype="string")
    )
    status = (
        result["Status"].astype("string").str.strip().str.lower()
        if "Status" in result.columns
        else pd.Series("", index=result.index, dtype="string")
    )

    missing_status = existing.isna() | (existing.str.strip() == "")

    needs_drawing = (
        work_type.eq("plan drafting")
        & status.isin(["work request", "work requested"])
        & missing_status
    )
    survey_ready = (
        work_type.eq("geospatial asset mapping")
        & missing_status
    )

    result.loc[needs_drawing, "Drawing Status"] = "Needs Drawing"
    result.loc[survey_ready, "Drawing Status"] = "Ready"

    return result


def prepare_master_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Drop completely empty rows/columns created by Salesforce report layout.
    result = result.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cleaned_columns = []
    for c in result.columns:
        text = _cell_text(c)
        text = re.sub(r"\s*↑\s*$", "", text).strip()
        text = re.sub(r"\s+", " ", text)
        cleaned_columns.append(text)
    result.columns = cleaned_columns

    # Remove any unnamed / blank header columns.
    keep_cols = [
        c for c in result.columns
        if c and not c.lower().startswith("unnamed:")
    ]
    result = result[keep_cols].copy()

    result = _forward_fill_grouped_salesforce_fields(result)
    result = derive_drawing_status(result)

    # The new Salesforce master export may not contain a standalone Postcode.
    if "Postcode" not in result.columns:
        result["Postcode"] = ""
    else:
        result["Postcode"] = result["Postcode"].fillna("").astype(str)

    if "Building Name" in result.columns:
        missing_postcode = result["Postcode"].str.strip().eq("")
        result.loc[missing_postcode, "Postcode"] = (
            result.loc[missing_postcode, "Building Name"]
            .apply(extract_postcode)
        )

    # Salesforce reports can include a final totals/footer row. Keep only rows
    # that actually identify a building or customer reference.
    reference_cols = [
        c for c in [
            "Customer Reference Code",
            "Customer Reference",
        ]
        if c in result.columns
    ]
    building_present = (
        result["Building Name"].notna()
        & result["Building Name"].astype(str).str.strip().ne("")
        if "Building Name" in result.columns
        else pd.Series(False, index=result.index)
    )
    reference_present = pd.Series(False, index=result.index)
    for c in reference_cols:
        reference_present = (
            reference_present
            | (
                result[c].notna()
                & result[c].astype(str).str.strip().ne("")
            )
        )
    result = result[building_present | reference_present].copy()

    return result.reset_index(drop=True)


def read_salesforce_or_standard_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, int]:
    """
    Read either:
      - a normal Excel table whose headers are on row 1; or
      - a Salesforce report containing title/filter metadata above the real table.

    Returns (dataframe, zero_based_header_row).
    """
    raw = pd.read_excel(
        io.BytesIO(file_bytes),
        header=None,
    )
    header_row = find_header_row(raw)

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        header=header_row,
    )
    df = prepare_master_dataframe(df)
    return df, header_row

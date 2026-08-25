from __future__ import annotations

"""
Coordinate-based planning rules.

USER-TUNABLE RULES
------------------
These constants are deliberately kept together at the top of this file so the
geographic behaviour can be adjusted without touching the rest of the scheduler.

The scheduler still uses Google Routes for the journeys that matter. Coordinates
are used first to create sensible geographic clusters and to avoid paying Google
to distinguish buildings that are already known to be very close together.
"""

import math
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# USER-TUNABLE COORDINATE CLUSTERING RULES
# ============================================================================

# Strategic weekly clusters are built with complete-linkage clustering. This
# value is the maximum straight-line span allowed inside one coordinate cluster.
# Complete linkage prevents the "chain" problem where a series of individually
# close buildings could otherwise create one huge cluster across London.
GEOGRAPHIC_CLUSTER_MAX_DIAMETER_KM = 2.5

# If two buildings are within this straight-line distance, the site-to-site move
# does not need its own Google Route Matrix destination. A cheap local walking
# estimate is used instead. Exact same full postcodes also use this shortcut.
NO_GOOGLE_RADIUS_KM = 0.40

# Very close sites are treated as campus-like for the local transfer estimate.
SAME_CAMPUS_RADIUS_KM = 0.10

# Future-work support radii used by the rolling-horizon/endgame logic. A current
# building close to future work is more valuable as an anchor for a later week.
FUTURE_ANCHOR_STRONG_RADIUS_KM = 0.50
FUTURE_ANCHOR_SUPPORT_RADIUS_KM = 1.25

# Local-transfer estimate used when Google is deliberately bypassed.
LOCAL_WALKING_SPEED_KMPH = 4.8
LOCAL_NAVIGATION_ALLOWANCE_MINUTES = 1.0

# Earth radius used by the Haversine calculation.
EARTH_RADIUS_KM = 6371.0088


# ============================================================================
# Coordinate helpers
# ============================================================================


def clean_coordinate(value, minimum: float, maximum: float) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < minimum or result > maximum:
        return None
    return result


def clean_latitude(value) -> Optional[float]:
    return clean_coordinate(value, -90.0, 90.0)


def clean_longitude(value) -> Optional[float]:
    return clean_coordinate(value, -180.0, 180.0)


def valid_coordinate_pair(latitude, longitude) -> bool:
    return (
        clean_latitude(latitude) is not None
        and clean_longitude(longitude) is not None
    )


def normalise_postcode(postcode: str) -> str:
    return re.sub(r"\s+", "", str(postcode or "").upper())


def postcode_district(postcode: str) -> str:
    text = str(postcode or "").strip().upper()
    if not text:
        return ""
    return text.split()[0]


def same_full_postcode(a: str, b: str) -> bool:
    pc_a = normalise_postcode(a)
    pc_b = normalise_postcode(b)
    return bool(pc_a) and pc_a == pc_b


def haversine_km(
    latitude_a,
    longitude_a,
    latitude_b,
    longitude_b,
) -> Optional[float]:
    lat_a = clean_latitude(latitude_a)
    lon_a = clean_longitude(longitude_a)
    lat_b = clean_latitude(latitude_b)
    lon_b = clean_longitude(longitude_b)
    if None in (lat_a, lon_a, lat_b, lon_b):
        return None

    phi1 = math.radians(lat_a)
    phi2 = math.radians(lat_b)
    dphi = math.radians(lat_b - lat_a)
    dlambda = math.radians(lon_b - lon_a)

    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def site_distance_km(site_a: dict, site_b: dict) -> Optional[float]:
    return haversine_km(
        site_a.get("latitude"),
        site_a.get("longitude"),
        site_b.get("latitude"),
        site_b.get("longitude"),
    )


def should_bypass_google_between_sites(
    site_a: dict,
    site_b: dict,
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> Tuple[bool, Optional[float], str]:
    """
    Return (bypass_google, straight_line_distance_km, reason).

    Exact same full postcode is an explicit user rule. Otherwise coordinates are
    used. If neither signal is available the old routing path can take over.
    """
    if same_full_postcode(site_a.get("postcode"), site_b.get("postcode")):
        distance = site_distance_km(site_a, site_b)
        return True, distance, "same full postcode"

    distance = site_distance_km(site_a, site_b)
    if distance is not None and distance <= float(radius_km):
        return True, distance, f"within {float(radius_km):.2f} km"

    return False, distance, ""


def estimate_local_transfer_minutes(
    distance_km: Optional[float],
    minimum_minutes: float = 5.0,
) -> float:
    """
    Conservative walking-style estimate for a deliberately un-routed local move.

    The existing UI transfer value remains the minimum, so changing coordinate
    clustering does not silently remove the user's existing site-transfer buffer.
    """
    minimum = max(0.0, float(minimum_minutes))
    if distance_km is None:
        return minimum
    if distance_km <= SAME_CAMPUS_RADIUS_KM:
        return minimum

    walking_minutes = (
        float(distance_km) / float(LOCAL_WALKING_SPEED_KMPH) * 60.0
        + float(LOCAL_NAVIGATION_ALLOWANCE_MINUTES)
    )
    return max(minimum, float(math.ceil(walking_minutes)))


# ============================================================================
# Portfolio clustering
# ============================================================================


def _project_to_local_km(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Cheap local equirectangular projection suitable for UK-scale clustering."""
    reference_lat = float(np.nanmedian(latitudes))
    x = (
        np.radians(longitudes)
        * EARTH_RADIUS_KM
        * math.cos(math.radians(reference_lat))
    )
    y = np.radians(latitudes) * EARTH_RADIUS_KM
    return np.column_stack([x, y])


def _dominant_district(values) -> str:
    districts = [postcode_district(v) for v in values]
    districts = [d for d in districts if d]
    if not districts:
        return "AREA"
    counts = pd.Series(districts).value_counts()
    return str(counts.index[0])


def assign_coordinate_planning_clusters(
    df: pd.DataFrame,
    max_diameter_km: float = GEOGRAPHIC_CLUSTER_MAX_DIAMETER_KM,
) -> pd.DataFrame:
    """
    Add coordinate-driven planning metadata to the full portfolio.

    - Rows with Latitude + Longitude are clustered by real straight-line distance.
    - Complete linkage keeps every pair of buildings in a cluster within roughly
      max_diameter_km, avoiding sprawling chain clusters.
    - Rows without coordinates fall back to the original postcode-district rule.
    - The existing ``Postcode Cluster`` field is retained for reference.
    - ``Planning Cluster`` is the field the strategic scheduler should use.
    """
    result = df.copy()

    if "Postcode" not in result.columns:
        result["Postcode"] = ""

    result["Postcode Cluster"] = result["Postcode"].apply(postcode_district)

    lat_source = result.get("Latitude", pd.Series(index=result.index, dtype=float))
    lon_source = result.get("Longitude", pd.Series(index=result.index, dtype=float))

    result["Latitude Clean"] = lat_source.apply(clean_latitude)
    result["Longitude Clean"] = lon_source.apply(clean_longitude)
    result["Coordinate Available"] = (
        result["Latitude Clean"].notna()
        & result["Longitude Clean"].notna()
    )

    result["Planning Cluster"] = result["Postcode Cluster"].astype(str)
    result["Geographic Cluster"] = ""
    result["Cluster Centre Latitude"] = np.nan
    result["Cluster Centre Longitude"] = np.nan
    result["Coordinate Cluster Source"] = "Postcode fallback"
    result["Nearby Portfolio Sites <= No-Google Radius"] = 0
    result["Nearby Portfolio Sites <= Anchor Radius"] = 0
    result["Nearest Portfolio Site (km)"] = np.nan

    valid = result[result["Coordinate Available"] == True].copy()
    if valid.empty:
        return result

    latitudes = valid["Latitude Clean"].astype(float).to_numpy()
    longitudes = valid["Longitude Clean"].astype(float).to_numpy()
    projected = _project_to_local_km(latitudes, longitudes)

    if len(valid) == 1:
        raw_labels = np.array([0], dtype=int)
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(max_diameter_km),
            linkage="complete",
            metric="euclidean",
            compute_full_tree=True,
        )
        raw_labels = model.fit_predict(projected)

    valid = valid.copy()
    valid["_raw_geo_cluster"] = raw_labels

    # Human-readable cluster labels use the dominant postcode district plus the
    # cluster centroid. The centroid makes distinct clusters in the same outward
    # postcode easy to distinguish without relying on postcode text for distance.
    label_map = {}
    centre_map = {}
    for raw_label, group in valid.groupby("_raw_geo_cluster"):
        centre_lat = float(group["Latitude Clean"].astype(float).mean())
        centre_lon = float(group["Longitude Clean"].astype(float).mean())
        district = _dominant_district(group["Postcode"].tolist())
        label = f"GEO-{district}-{centre_lat:.3f}_{centre_lon:.3f}"
        label_map[int(raw_label)] = label
        centre_map[int(raw_label)] = (centre_lat, centre_lon)

    for idx, raw_label in zip(valid.index, raw_labels):
        label = label_map[int(raw_label)]
        centre_lat, centre_lon = centre_map[int(raw_label)]
        result.at[idx, "Planning Cluster"] = label
        result.at[idx, "Geographic Cluster"] = label
        result.at[idx, "Cluster Centre Latitude"] = centre_lat
        result.at[idx, "Cluster Centre Longitude"] = centre_lon
        result.at[idx, "Coordinate Cluster Source"] = "Latitude/Longitude"

    # Cheap portfolio-density signals used by shortlist and endgame logic.
    neighbours_micro = NearestNeighbors(
        radius=float(NO_GOOGLE_RADIUS_KM),
        metric="euclidean",
    ).fit(projected)
    micro_lists = neighbours_micro.radius_neighbors(
        projected,
        return_distance=False,
    )

    neighbours_anchor = NearestNeighbors(
        radius=float(FUTURE_ANCHOR_SUPPORT_RADIUS_KM),
        metric="euclidean",
    ).fit(projected)
    anchor_lists = neighbours_anchor.radius_neighbors(
        projected,
        return_distance=False,
    )

    if len(valid) > 1:
        nearest_model = NearestNeighbors(
            n_neighbors=2,
            metric="euclidean",
        ).fit(projected)
        nearest_distances, _ = nearest_model.kneighbors(projected)
        nearest_values = nearest_distances[:, 1]
    else:
        nearest_values = np.array([np.nan])

    for pos, idx in enumerate(valid.index):
        result.at[idx, "Nearby Portfolio Sites <= No-Google Radius"] = max(
            0,
            len(micro_lists[pos]) - 1,
        )
        result.at[idx, "Nearby Portfolio Sites <= Anchor Radius"] = max(
            0,
            len(anchor_lists[pos]) - 1,
        )
        result.at[idx, "Nearest Portfolio Site (km)"] = (
            round(float(nearest_values[pos]), 3)
            if math.isfinite(float(nearest_values[pos]))
            else np.nan
        )

    return result

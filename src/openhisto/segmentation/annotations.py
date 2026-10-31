"""GeoJSON/QuPath semantic annotations in an explicit pixel coordinate frame."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import MaskSpec


def _property(properties, name):
    value = properties
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_features(source):
    if isinstance(source, (str, Path)):
        with Path(source).open() as handle:
            source = json.load(handle)
    if isinstance(source, list):
        features = source
    elif isinstance(source, dict) and source.get("type") == "FeatureCollection":
        features = source.get("features", [])
    elif isinstance(source, dict) and source.get("type") == "Feature":
        features = [source]
    elif isinstance(source, dict) and source.get("type") in {"Polygon", "MultiPolygon"}:
        features = [{"type": "Feature", "geometry": source, "properties": {}}]
    else:
        raise ValueError("Annotations must be GeoJSON Polygon/MultiPolygon features")
    return features


def rasterize_geojson(
    source,
    shape,
    spec=None,
    *,
    xy_level0=None,
    footprint_level0=None,
    num_classes=2,
    ignore_index=-100,
):
    """Sample pixel centers. Later features win; first class_priority entry wins."""
    from shapely.geometry import shape as geometry_shape
    from shapely import contains_xy

    spec = spec or MaskSpec(kind="geojson")
    h, w = shape
    origin = xy_level0 if xy_level0 is not None else (0.0, 0.0)
    footprint = footprint_level0 if footprint_level0 is not None else (float(w), float(h))
    sx, sy = footprint[0] / w, footprint[1] / h
    result = np.zeros((h, w), dtype=np.int64)
    occupied = np.zeros((h, w), dtype=bool)
    parsed = []
    for feature in load_features(source):
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Only Polygon and MultiPolygon annotations are supported")
        poly = geometry_shape(geometry)
        if not poly.is_valid:
            raise ValueError("Annotation contains an invalid polygon")
        if poly.is_empty:
            continue
        name = _property(feature.get("properties", {}), spec.class_property)
        if spec.class_map:
            mapping = {str(k): int(v) for k, v in spec.class_map.items()}
            if str(name) not in mapping:
                raise ValueError(f"Unknown annotation class: {name!r}")
            class_id = mapping[str(name)]
        else:
            class_id = 1
        if class_id != ignore_index and not 0 <= class_id < num_classes:
            raise ValueError(f"Annotation class ID {class_id} is outside class range")
        parsed.append((class_id, poly))
    if spec.class_priority:
        rank = {int(v): i for i, v in enumerate(spec.class_priority)}
        missing = {c for c, _ in parsed} - set(rank)
        if missing:
            raise ValueError(f"class_priority must include annotation classes {sorted(missing)}")
        parsed.sort(key=lambda pair: rank[pair[0]], reverse=True)
    for class_id, poly in parsed:
        minx, miny, maxx, maxy = poly.bounds
        x0 = max(0, int(np.floor((minx - origin[0]) / sx)))
        y0 = max(0, int(np.floor((miny - origin[1]) / sy)))
        x1 = min(w, int(np.ceil((maxx - origin[0]) / sx)))
        y1 = min(h, int(np.ceil((maxy - origin[1]) / sy)))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = origin[0] + (np.arange(x0, x1) + 0.5) * sx
        ys = origin[1] + (np.arange(y0, y1) + 0.5) * sy
        inside = contains_xy(poly, xs[None, :], ys[:, None])
        region = result[y0:y1, x0:x1]
        used = occupied[y0:y1, x0:x1]
        region[inside] = class_id
        used[inside] = True
    return result


def labels_to_geojson(
    labels, *, xy_level0=None, footprint_level0=None, class_names=None, qupath=False
):
    """Trace exact pixel-cell boundaries, retaining holes and disconnected regions."""
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    labels = np.asarray(labels)
    h, w = labels.shape
    origin = xy_level0 if xy_level0 is not None else (0.0, 0.0)
    footprint = footprint_level0 if footprint_level0 is not None else (float(w), float(h))
    sx, sy = footprint[0] / w, footprint[1] / h
    features = []
    # Merge row runs instead of constructing one polygon for every foreground pixel.
    for class_id in sorted(int(x) for x in np.unique(labels) if x > 0):
        rectangles = []
        for y in range(h):
            row = labels[y] == class_id
            changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            for x0, x1 in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                rectangles.append(
                    box(
                        origin[0] + x0 * sx,
                        origin[1] + y * sy,
                        origin[0] + x1 * sx,
                        origin[1] + (y + 1) * sy,
                    )
                )
        merged = unary_union(rectangles)
        parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        name = (
            class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        )
        for polygon in parts:
            properties = {"class_id": class_id, "classification": {"name": name}}
            if qupath:
                properties["objectType"] = "annotation"
            features.append(
                {"type": "Feature", "geometry": mapping(polygon), "properties": properties}
            )
    return {"type": "FeatureCollection", "features": features}

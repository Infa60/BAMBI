import re
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional


def parse_run_id(run_id: str):
    """
    Normalize a run id and extract key tokens.

    Returns:
        root: 'BAMBI###' (uppercased)
        supine_num: integer supine number if present, else None
        letter: sub-run letter '', 'A'..'Z' (last single-letter segment)
        token: normalized id without the trailing 'is running'
    """
    token = run_id.split()[0]  # strip trailing "is running" if present

    # Root: allow underscore or end after digits (because '_' is a word char; \b won't work before '_').
    m_root = re.match(r'^(BAMBI\d+)(?:_|$)', token, flags=re.IGNORECASE)
    if not m_root:
        raise ValueError(f"Invalid run id (no BAMBI### prefix): {run_id}")
    root = m_root.group(1).upper()

    # Supine number: require start-or-underscore before 'supine' and underscore-or-end after the digits.
    m_sup = re.search(r'(?:^|_)supine[^0-9]*([0-9]+)(?:_|$)', token, flags=re.IGNORECASE)
    supine_num = int(m_sup.group(1)) if m_sup else None

    # Last single-letter segment delimited by underscores: _A, _B, ... (avoid LH/MC/NDB/3M)
    letters = re.findall(r'_(?P<letter>[A-Z])(?:_|$)', token, flags=re.IGNORECASE)
    letter = letters[-1].upper() if letters else ''

    return root, supine_num, letter, token


def group_runs_by_subject(data_by_run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group runs by subject root (BAMBI###).
    """
    grouped = defaultdict(dict)
    for run_id, data in data_by_run.items():
        root, _, _, token = parse_run_id(run_id)
        grouped[root][token] = data
    return grouped


def get_per_marker_dict(run_data: Any) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Normalize run data shape:
      - Accepts {marker: {metric: array}} or {'per_marker': {marker: {metric: array}}}
      - Returns the {marker: {metric: array}} view.
    """
    if isinstance(run_data, dict) and isinstance(run_data.get("per_marker"), dict):
        return run_data["per_marker"]
    return run_data


# -----------------------------------------------------------
# Case-insensitive key resolution (NEW)
# -----------------------------------------------------------

def resolve_key_ci(d: Dict[str, Any], key: str) -> Optional[str]:
    """
    Case-insensitive lookup of 'key' in dict 'd'.
    Returns the actual key if found, else None.
    """
    if key in d:
        return key
    kcf = key.casefold()
    for k in d.keys():
        if isinstance(k, str) and k.casefold() == kcf:
            return k
    return None


# -----------------------------------------------------------
# Sorting helpers
# -----------------------------------------------------------

def letter_rank(letter: str) -> int:
    """ '' < 'A' < 'B' < ... < 'Z' """
    return -1 if letter == '' else (ord(letter.upper()) - ord('A') + 1)

def sort_key_all(token: str) -> Tuple[int, int, str]:
    """ Global order: Supine number first, then sub-run letter. """
    _, sup, letter, _ = parse_run_id(token)
    sup_rank = sup if sup is not None else 10**9
    return (sup_rank, letter_rank(letter), token.lower())

def sort_key_within_supine(token: str) -> Tuple[int, str]:
    """ Inside a single Supine: '' < A < B < C < ... """
    _, _, letter, _ = parse_run_id(token)
    return (letter_rank(letter), token.lower())


# -----------------------------------------------------------
# Main concatenation (uses case-insensitive key resolution)
# -----------------------------------------------------------

def concat_marker_metric_for_subject(
    data_by_run: Dict[str, Any],
    subject_root: str,             # e.g., "BAMBI051"
    marker: str,                   # e.g., "RANK"
    metric: str,                   # e.g., "velocity"
    mode: str = "subject",         # "subject" or "subject_supine"
    include_pattern: Optional[str] = None
) -> Any:
    """
    Time-wise concatenation (axis=0) of a given marker/metric across runs of one subject.

    Notes:
      * Accepts 1D arrays (T,) and 2D arrays (T, F). 1D arrays are reshaped to (T, 1).
      * Ensures consistent number of columns F across concatenated runs; raises ValueError otherwise.
      * If include_pattern is provided, only run ids matching the regex are considered.
      * Marker/metric lookup is case-insensitive.
    """
    grouped = group_runs_by_subject(data_by_run)
    subject_runs = grouped.get(subject_root.upper(), {})
    if not subject_runs:
        raise KeyError(f"No runs found for subject {subject_root}")

    # Optional filter on run id token
    if include_pattern:
        rgx = re.compile(include_pattern)
        run_ids = [rid for rid in subject_runs.keys() if rgx.search(rid)]
    else:
        run_ids = list(subject_runs.keys())

    if mode == "subject":
        run_ids.sort(key=sort_key_all)

        blocks, segments, runs_used = [], [], []
        F_ref = None

        for rid in run_ids:
            per_marker = get_per_marker_dict(subject_runs[rid])

            mk = resolve_key_ci(per_marker, marker)
            if mk is None:
                continue
            met = resolve_key_ci(per_marker[mk], metric)
            if met is None:
                continue

            arr = np.asarray(per_marker[mk][met])
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim != 2:
                raise ValueError(f"Unsupported shape for {rid}:{marker}:{metric} -> {arr.shape}")

            F = arr.shape[1]
            if F_ref is None:
                F_ref = F
            elif F != F_ref:
                raise ValueError(f"Inconsistent number of columns across runs: expected {F_ref}, got {F} in {rid}")

            blocks.append(arr)
            segments.append({'run_id': rid, 'len': arr.shape[0]})
            runs_used.append(rid)

        if not blocks:
            raise ValueError(f"No data for {subject_root}/{marker}/{metric} (mode=subject)")
        return {'array': np.concatenate(blocks, axis=0), 'segments': segments, 'runs_used': runs_used}

    elif mode == "subject_supine":
        buckets = defaultdict(list)  # supine_num -> [token, ...]
        for rid in run_ids:
            _, sup, _, _ = parse_run_id(rid)
            if sup is not None:
                buckets[sup].append(rid)

        results = {}
        for sup in sorted(buckets.keys()):
            rids = sorted(buckets[sup], key=sort_key_within_supine)

            blocks, segments, runs_used = [], [], []
            F_ref = None

            for rid in rids:
                per_marker = get_per_marker_dict(subject_runs[rid])

                mk = resolve_key_ci(per_marker, marker)
                if mk is None:
                    continue
                met = resolve_key_ci(per_marker[mk], metric)
                if met is None:
                    continue

                arr = np.asarray(per_marker[mk][met])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim != 2:
                    raise ValueError(f"Unsupported shape for {rid}:{marker}:{metric} -> {arr.shape}")

                F = arr.shape[1]
                if F_ref is None:
                    F_ref = F
                elif F != F_ref:
                    raise ValueError(f"Inconsistent columns in Supine{sup}: expected {F_ref}, got {F} in {rid}")

                blocks.append(arr)
                segments.append({'run_id': rid, 'len': arr.shape[0]})
                runs_used.append(rid)

            if blocks:
                results[sup] = {'array': np.concatenate(blocks, axis=0),
                                'segments': segments,
                                'runs_used': runs_used}

        if not results:
            raise ValueError(f"No per-supine data for {subject_root}/{marker}/{metric} (mode=subject_supine)")
        return results

    else:
        raise ValueError("mode must be 'subject' or 'subject_supine'")
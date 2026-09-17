"""Conservative, transitive grouping of related source images before splitting."""
from __future__ import annotations

import hashlib
from pathlib import PureWindowsPath


def source_key(file_name: str) -> str:
    name = PureWindowsPath(file_name).name.casefold()
    # Roboflow appends a unique suffix to each exported/augmented variant.
    return name.split(".rf.", 1)[0] if ".rf." in name else PureWindowsPath(name).stem


def assign_groups(records: list[dict], use_perceptual: bool = True) -> None:
    """Union source names, exact hashes and optional perceptual matches.

    Keep aliases from removed duplicates: A/B with identical bytes and B/C with
    a common source must form one group, even when A is the retained record.
    Source names are conservative hints, not proof of identical photographs.
    """
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        sources = record.get("source_keys") or [source_key(record["source_file_name"])]
        record["source_keys"] = sorted(set(sources))
        keys = [("source", value) for value in sources]
        keys.append(("sha256", record["sha256"]))
        if use_perceptual and record.get("perceptual_group"):
            keys.append(("perceptual", record["perceptual_group"]))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    members: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        members.setdefault(root(index), []).append(record["sha256"])
    identifiers = {
        key: hashlib.sha256("\n".join(sorted(set(values))).encode()).hexdigest()
        for key, values in members.items()
    }
    for index, record in enumerate(records):
        record["split_group"] = identifiers[root(index)]


def split_overlap_report(splits: dict[str, list[dict]]) -> dict:
    report = {}
    for field in ("sha256", "source_keys", "perceptual_group", "split_group"):
        sets = {}
        for name, records in splits.items():
            values = set()
            for record in records:
                value = record.get(field)
                values.update(value if isinstance(value, list) else [value] if value else [])
            sets[name] = values
        names = list(sets)
        report[field] = {
            f"{left}/{right}": len(sets[left] & sets[right])
            for i, left in enumerate(names) for right in names[i + 1:]
        }
    return report

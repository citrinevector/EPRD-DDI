#!/usr/bin/env python3
"""Audit a DDI train/validation/test split for a transductive experiment.

Core checks
-----------
1. Required columns and valid numeric labels.
2. Duplicate or conflicting ordered drug pairs within each split.
3. Exact ordered-pair overlap across splits.
4. Reversed-pair overlap across splits (failed by default because it can leak
   the same molecular pair; use --allow-reversed-across-splits only after a
   documented review of directional labels).
5. Transductive drug coverage:
       V_validation union V_test is a subset of V_train.
6. Validation/test label coverage by the training set.
7. Expected number and indexing pattern of DDI classes.
8. Optional exact verification of the per-class allocation implemented by
   build_transductive_86.py.

Example
-------
python verify_transductive_split.py \
  --split-dir ./datasets/drugbank/0/transductive_strict \
  --num-classes 86 \
  --check-builder-allocation \
  --show-class-counts \
  --json-out ./logs/transductive_split_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("drug_1", "drug_2", "label")
SPLIT_FILENAMES = {
    "train": "ddi_training.csv",
    "validation": "ddi_validation.csv",
    "test": "ddi_test.csv",
}


def _examples(items: list[Any] | set[Any], limit: int) -> list[Any]:
    ordered = sorted(items, key=lambda value: repr(value))
    output: list[Any] = []
    for item in ordered[:limit]:
        if isinstance(item, tuple):
            output.append(list(item))
        else:
            output.append(item)
    return output


def _canonical_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted((drug_a, drug_b)))


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{split_name}: file not found: {path}")

    frame = pd.read_csv(path, dtype={"drug_1": "string", "drug_2": "string"})
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{split_name}: missing required column(s): {', '.join(missing_columns)}"
        )

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    if frame.isna().any().any():
        missing = frame.isna().sum()
        details = ", ".join(f"{column}={int(count)}" for column, count in missing.items() if count)
        raise ValueError(f"{split_name}: missing values detected ({details})")

    for column in ("drug_1", "drug_2"):
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"{split_name}: empty drug identifier(s) in {column}")

    numeric_labels = pd.to_numeric(frame["label"], errors="raise")
    if ((numeric_labels % 1) != 0).any():
        raise ValueError(f"{split_name}: labels must be integers")
    frame["label"] = numeric_labels.astype(int)
    return frame


def audit_splits(
    frames: dict[str, pd.DataFrame],
    num_classes: int,
    allow_reversed_across_splits: bool = False,
    max_examples: int = 10,
    require_validation_all_classes: bool = False,
    check_builder_allocation: bool = False,
    test_ratio: float = 0.20,
    validation_ratio: float = 0.10,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    pair_sets: dict[str, set[tuple[str, str]]] = {}
    canonical_sets: dict[str, set[tuple[str, str]]] = {}
    drug_sets: dict[str, set[str]] = {}
    label_sets: dict[str, set[int]] = {}
    class_counts: dict[str, dict[int, int]] = {}
    split_stats: dict[str, dict[str, Any]] = {}

    for name, frame in frames.items():
        ordered_pairs = list(zip(frame["drug_1"], frame["drug_2"]))
        pair_sets[name] = set(ordered_pairs)
        canonical_sets[name] = {
            _canonical_pair(drug_a, drug_b) for drug_a, drug_b in ordered_pairs
        }
        drug_sets[name] = set(frame["drug_1"]) | set(frame["drug_2"])
        label_sets[name] = set(int(label) for label in frame["label"])
        counts = frame["label"].value_counts().sort_index()
        class_counts[name] = {int(label): int(count) for label, count in counts.items()}

        duplicate_mask = frame.duplicated(["drug_1", "drug_2"], keep=False)
        duplicate_rows = frame.loc[duplicate_mask]
        if not duplicate_rows.empty:
            duplicate_pairs = set(zip(duplicate_rows["drug_1"], duplicate_rows["drug_2"]))
            conflicting = (
                duplicate_rows.groupby(["drug_1", "drug_2"])["label"].nunique() > 1
            )
            conflict_pairs = set(conflicting[conflicting].index.tolist())
            errors.append(
                f"{name}: {len(duplicate_pairs)} ordered pair(s) occur more than once; "
                f"examples={_examples(duplicate_pairs, max_examples)}"
            )
            if conflict_pairs:
                errors.append(
                    f"{name}: {len(conflict_pairs)} ordered pair(s) have conflicting labels; "
                    f"examples={_examples(conflict_pairs, max_examples)}"
                )

        self_pairs = {(a, b) for a, b in pair_sets[name] if a == b}
        if self_pairs:
            warnings.append(
                f"{name}: {len(self_pairs)} self-pair(s) detected; "
                f"examples={_examples(self_pairs, max_examples)}"
            )

        reversed_within = {
            _canonical_pair(a, b)
            for a, b in pair_sets[name]
            if a != b and (b, a) in pair_sets[name]
        }
        if reversed_within:
            warnings.append(
                f"{name}: both orientations occur for {len(reversed_within)} molecular pair(s); "
                f"review whether the directional labels are intentional; "
                f"examples={_examples(reversed_within, max_examples)}"
            )

        split_stats[name] = {
            "rows": int(len(frame)),
            "unique_ordered_pairs": int(len(pair_sets[name])),
            "unique_unordered_pairs": int(len(canonical_sets[name])),
            "drugs": int(len(drug_sets[name])),
            "labels": int(len(label_sets[name])),
            "minimum_class_support": int(counts.min()) if not counts.empty else 0,
            "maximum_class_support": int(counts.max()) if not counts.empty else 0,
        }

    cross_split: dict[str, Any] = {}
    for left, right in combinations(("train", "validation", "test"), 2):
        key = f"{left}__{right}"
        exact_overlap = pair_sets[left] & pair_sets[right]
        reversed_overlap = {
            (a, b)
            for a, b in pair_sets[left]
            if a != b and (b, a) in pair_sets[right]
        }
        cross_split[key] = {
            "exact_ordered_overlap_count": len(exact_overlap),
            "exact_ordered_overlap_examples": _examples(exact_overlap, max_examples),
            "reversed_overlap_count": len(reversed_overlap),
            "reversed_overlap_examples_from_first_split": _examples(
                reversed_overlap, max_examples
            ),
        }

        if exact_overlap:
            errors.append(
                f"{left}/{right}: {len(exact_overlap)} exact ordered pair(s) overlap; "
                f"examples={_examples(exact_overlap, max_examples)}"
            )
        else:
            passes.append(f"{left}/{right}: no exact ordered-pair overlap")

        if reversed_overlap:
            message = (
                f"{left}/{right}: {len(reversed_overlap)} reversed molecular pair(s) cross the split "
                f"boundary; examples from {left}={_examples(reversed_overlap, max_examples)}"
            )
            if allow_reversed_across_splits:
                warnings.append(message)
            else:
                errors.append(message)
        else:
            passes.append(f"{left}/{right}: no reversed-pair overlap")

    unseen_validation_drugs = drug_sets["validation"] - drug_sets["train"]
    unseen_test_drugs = drug_sets["test"] - drug_sets["train"]
    if unseen_validation_drugs:
        errors.append(
            f"validation: {len(unseen_validation_drugs)} drug(s) are absent from training; "
            f"examples={_examples(unseen_validation_drugs, max_examples)}"
        )
    else:
        passes.append("validation drug set is a subset of the training drug set")
    if unseen_test_drugs:
        errors.append(
            f"test: {len(unseen_test_drugs)} drug(s) are absent from training; "
            f"examples={_examples(unseen_test_drugs, max_examples)}"
        )
    else:
        passes.append("test drug set is a subset of the training drug set")

    unseen_validation_labels = label_sets["validation"] - label_sets["train"]
    unseen_test_labels = label_sets["test"] - label_sets["train"]
    if unseen_validation_labels:
        errors.append(
            f"validation: label(s) absent from training: "
            f"{_examples(unseen_validation_labels, max_examples)}"
        )
    else:
        passes.append("all validation labels occur in training")
    if unseen_test_labels:
        errors.append(
            f"test: label(s) absent from training: {_examples(unseen_test_labels, max_examples)}"
        )
    else:
        passes.append("all test labels occur in training")

    all_labels = set().union(*label_sets.values())
    if len(label_sets["train"]) != num_classes:
        errors.append(
            f"training contains {len(label_sets['train'])} classes, expected {num_classes}; "
            f"labels={_examples(label_sets['train'], max_examples)}"
        )
    else:
        passes.append(f"training contains all {num_classes} expected classes")
    if len(all_labels) != num_classes:
        errors.append(
            f"the complete split contains {len(all_labels)} classes, expected {num_classes}"
        )
    else:
        passes.append(f"the complete split contains {num_classes} classes")

    zero_based = set(range(num_classes))
    one_based = set(range(1, num_classes + 1))
    if all_labels == zero_based:
        passes.append(f"class identifiers are contiguous and zero-based (0-{num_classes - 1})")
        label_indexing = "zero_based"
    elif all_labels == one_based:
        passes.append(f"class identifiers are contiguous and one-based (1-{num_classes})")
        label_indexing = "one_based"
    else:
        warnings.append(
            "class identifiers are not a complete contiguous zero-based or one-based range; "
            "verify the label mapping"
        )
        label_indexing = "non_contiguous_or_incomplete"

    expected_labels = (
        zero_based
        if label_indexing == "zero_based"
        else one_based
        if label_indexing == "one_based"
        else all_labels
    )
    missing_by_split = {
        name: expected_labels - label_sets[name]
        for name in ("train", "validation", "test")
    }
    if not missing_by_split["test"]:
        passes.append(f"test contains all {num_classes} expected classes")
    else:
        errors.append(
            f"test is missing {len(missing_by_split['test'])} expected class(es); "
            f"labels={_examples(missing_by_split['test'], max_examples)}"
        )

    if not missing_by_split["validation"]:
        passes.append(f"validation contains all {num_classes} expected classes")
    else:
        message = (
            f"validation is missing {len(missing_by_split['validation'])} expected class(es); "
            f"labels={_examples(missing_by_split['validation'], max_examples)}"
        )
        if require_validation_all_classes:
            errors.append(message)
        else:
            warnings.append(message)

    builder_allocation: dict[str, Any] = {
        "checked": check_builder_allocation,
        "test_ratio": test_ratio,
        "validation_ratio": validation_ratio,
        "mismatch_count": 0,
        "mismatch_examples": [],
    }
    if check_builder_allocation:
        allocation_mismatches: list[dict[str, Any]] = []
        for label in sorted(all_labels):
            total = sum(class_counts[name].get(label, 0) for name in frames)
            if total == 1:
                expected = {"train": 0, "validation": 0, "test": 1}
            elif total == 2:
                expected = {"train": 1, "validation": 0, "test": 1}
            else:
                expected_test = max(1, round(total * test_ratio))
                expected_validation = max(1, round(total * validation_ratio))
                if expected_test + expected_validation >= total:
                    expected_test, expected_validation = 1, 1
                expected = {
                    "train": total - expected_test - expected_validation,
                    "validation": expected_validation,
                    "test": expected_test,
                }
            actual = {
                name: class_counts[name].get(label, 0)
                for name in ("train", "validation", "test")
            }
            if actual != expected:
                allocation_mismatches.append(
                    {
                        "label": int(label),
                        "total": int(total),
                        "expected": expected,
                        "actual": actual,
                    }
                )

        builder_allocation["mismatch_count"] = len(allocation_mismatches)
        builder_allocation["mismatch_examples"] = allocation_mismatches[:max_examples]
        if allocation_mismatches:
            errors.append(
                "per-class counts do not match build_transductive_86.py for "
                f"{len(allocation_mismatches)} class(es); "
                f"examples={allocation_mismatches[:max_examples]}"
            )
        else:
            passes.append(
                "per-class counts exactly match build_transductive_86.py "
                f"(test_ratio={test_ratio}, validation_ratio={validation_ratio})"
            )

    total_rows = sum(len(frame) for frame in frames.values())
    split_stats["row_ratios"] = {
        name: (len(frame) / total_rows if total_rows else 0.0)
        for name, frame in frames.items()
    }

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "passes": passes,
        "split_statistics": split_stats,
        "class_counts": class_counts,
        "cross_split_overlap": cross_split,
        "transductive_coverage": {
            "unseen_validation_drugs_count": len(unseen_validation_drugs),
            "unseen_validation_drug_examples": _examples(
                unseen_validation_drugs, max_examples
            ),
            "unseen_test_drugs_count": len(unseen_test_drugs),
            "unseen_test_drug_examples": _examples(unseen_test_drugs, max_examples),
        },
        "label_coverage": {
            "expected_class_count": num_classes,
            "training_class_count": len(label_sets["train"]),
            "overall_class_count": len(all_labels),
            "indexing": label_indexing,
            "missing_by_split": {
                name: _examples(labels, max_examples)
                for name, labels in missing_by_split.items()
            },
            "unseen_validation_labels": _examples(
                unseen_validation_labels, max_examples
            ),
            "unseen_test_labels": _examples(unseen_test_labels, max_examples),
        },
        "builder_allocation": builder_allocation,
    }


def print_report(report: dict[str, Any], show_class_counts: bool = False) -> None:
    print("\n=== Split statistics ===")
    for name in ("train", "validation", "test"):
        stats = report["split_statistics"][name]
        print(
            f"{name:>10}: rows={stats['rows']:,}, "
            f"ordered_pairs={stats['unique_ordered_pairs']:,}, "
            f"drugs={stats['drugs']:,}, labels={stats['labels']}, "
            f"class_support=[{stats['minimum_class_support']}, "
            f"{stats['maximum_class_support']}]"
        )

    ratios = report["split_statistics"]["row_ratios"]
    print(
        "row ratios: "
        + ", ".join(f"{name}={ratio:.4f}" for name, ratio in ratios.items())
    )

    if show_class_counts:
        print("\n=== Per-class counts ===")
        labels = sorted(
            set().union(*(set(counts) for counts in report["class_counts"].values()))
        )
        print(f"{'label':>8} {'train':>10} {'validation':>12} {'test':>10}")
        for label in labels:
            print(
                f"{label:>8} "
                f"{report['class_counts']['train'].get(label, 0):>10} "
                f"{report['class_counts']['validation'].get(label, 0):>12} "
                f"{report['class_counts']['test'].get(label, 0):>10}"
            )

    print("\n=== Passed checks ===")
    for message in report["passes"]:
        print(f"[PASS] {message}")

    if report["warnings"]:
        print("\n=== Warnings requiring review ===")
        for message in report["warnings"]:
            print(f"[WARN] {message}")

    if report["errors"]:
        print("\n=== Failed checks ===")
        for message in report["errors"]:
            print(f"[FAIL] {message}")

    print(f"\nFINAL STATUS: {report['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that DrugBank DDI CSV files satisfy a strict transductive split."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("./datasets/drugbank/0/transductive_strict"),
        help="Directory containing ddi_training.csv, ddi_validation.csv, and ddi_test.csv.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=86,
        help="Expected number of event classes (default: 86).",
    )
    parser.add_argument(
        "--allow-reversed-across-splits",
        action="store_true",
        help=(
            "Downgrade reversed-pair overlap across splits from failure to warning. "
            "Use only after confirming that opposite orientations are intentionally distinct."
        ),
    )
    parser.add_argument(
        "--require-validation-all-classes",
        action="store_true",
        help=(
            "Treat missing validation classes as a failure. By default they are a warning; "
            "training and test coverage remain mandatory."
        ),
    )
    parser.add_argument(
        "--check-builder-allocation",
        action="store_true",
        help=(
            "Require every class count to match the allocation logic in "
            "build_transductive_86.py."
        ),
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="TEST_RATIO used by build_transductive_86.py (default: 0.20).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="VAL_RATIO used by build_transductive_86.py (default: 0.10).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum examples displayed for each issue (default: 10).",
    )
    parser.add_argument(
        "--show-class-counts",
        action="store_true",
        help="Print the count of every label in each split.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON audit report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not 0 < args.test_ratio < 1:
            raise ValueError("--test-ratio must be between 0 and 1")
        if not 0 < args.validation_ratio < 1:
            raise ValueError("--validation-ratio must be between 0 and 1")
        if args.test_ratio + args.validation_ratio >= 1:
            raise ValueError("--test-ratio + --validation-ratio must be less than 1")
        frames = {
            name: load_split(args.split_dir / filename, name)
            for name, filename in SPLIT_FILENAMES.items()
        }
        report = audit_splits(
            frames,
            num_classes=args.num_classes,
            allow_reversed_across_splits=args.allow_reversed_across_splits,
            max_examples=args.max_examples,
            require_validation_all_classes=args.require_validation_all_classes,
            check_builder_allocation=args.check_builder_allocation,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"[FAIL] Unable to audit split: {exc}", file=sys.stderr)
        return 2

    print_report(report, show_class_counts=args.show_class_counts)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"JSON report written to: {args.json_out}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the strict-transductive split of the 86-event DrugBank benchmark.

The three deduplicated files in datasets/drugbank/0/clean_v1/ are merged into one pool.
Pairs are allocated per event class (20% test, 10% validation, remainder training;
split seed 42). Same-class swaps between held-out and training pairs then ensure that
every validation and test drug also occurs in training, without changing the subset
sizes or the per-class counts. The split is written to
datasets/drugbank/0/transductive_strict/; the source files are not modified.

Usage:
    python datasets/build_transductive_86.py
"""
import os
from collections import Counter, defaultdict

import pandas as pd

NUM_CLASSES = 86
TEST_RATIO = 0.20  # fraction of each class assigned to the test set
VAL_RATIO = 0.10   # fraction of each class assigned to the validation set
SEED = 42

# datasets/drugbank/0, relative to this script
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drugbank", "0")

SRC = "clean_v1"             # deduplicated source files (training, validation, test)
OUT = "transductive_strict"  # output directory read by train.py


def load(part):
    path = os.path.join(DATA_ROOT, SRC, f"ddi_{part}.csv")
    return pd.read_csv(path)


def repair_transductive(all_data, train_df, val_df, test_df):
    """Swap held-out and training pairs of the same class until every drug occurs in training."""
    row_table = all_data.set_index("_row_id", drop=False)

    split_ids = {
        "train": set(train_df["_row_id"].astype(int)),
        "validation": set(val_df["_row_id"].astype(int)),
        "test": set(test_df["_row_id"].astype(int)),
    }

    def endpoints(row_id):
        d1 = row_table.at[row_id, "drug_1"]
        d2 = row_table.at[row_id, "drug_2"]
        return (d1,) if d1 == d2 else (d1, d2)

    labels = row_table["label"].astype(int).to_dict()

    # Number of training pairs that contain each drug
    degree = Counter()
    for row_id in split_ids["train"]:
        degree.update(endpoints(row_id))

    all_drugs = set(all_data["drug_1"]) | set(all_data["drug_2"])
    initially_missing = sorted(all_drugs - set(degree))

    train_by_label = defaultdict(set)
    for row_id in split_ids["train"]:
        train_by_label[labels[row_id]].add(row_id)

    swap_count = 0

    while True:
        missing = all_drugs - set(degree)
        if not missing:
            break

        options = []

        for split_priority, split_name in enumerate(("validation", "test")):
            for heldout_id in sorted(split_ids[split_name]):
                newly_covered = set(endpoints(heldout_id)) & missing
                if not newly_covered:
                    continue

                label = labels[heldout_id]
                best_training = None

                # Only training pairs with the same label are considered
                for training_id in sorted(train_by_label[label]):
                    training_drugs = endpoints(training_id)

                    # Both drugs of the outgoing training pair must remain in training
                    if not all(degree[drug] >= 2 for drug in training_drugs):
                        continue

                    remaining = [
                        degree[drug] - 1 for drug in training_drugs
                    ]
                    safety = (
                        min(remaining),
                        sum(remaining),
                        -training_id,
                    )

                    if best_training is None or safety > best_training[0]:
                        best_training = (safety, training_id)

                if best_training is None:
                    continue

                safety, training_id = best_training
                score = (
                    len(newly_covered),
                    safety[0],
                    safety[1],
                    -split_priority,
                    -heldout_id,
                )
                options.append(
                    (score, split_name, heldout_id, training_id)
                )

        if not options:
            raise RuntimeError(
                "Same-class swaps cannot place every drug in the training set; "
                f"drugs still missing: {sorted(missing)}"
            )

        _, split_name, heldout_id, training_id = max(
            options, key=lambda x: x[0]
        )
        label = labels[heldout_id]

        # Update the training degrees
        for drug in endpoints(training_id):
            degree[drug] -= 1
        for drug in endpoints(heldout_id):
            degree[drug] += 1

        # Swap one held-out pair with one training pair of the same class
        split_ids["train"].remove(training_id)
        split_ids["train"].add(heldout_id)

        split_ids[split_name].remove(heldout_id)
        split_ids[split_name].add(training_id)

        train_by_label[label].remove(training_id)
        train_by_label[label].add(heldout_id)

        swap_count += 1

    # The internal _row_id column is not written to the output files
    original_columns = [
        column for column in all_data.columns
        if column != "_row_id"
    ]

    def materialize(ids):
        return (
            all_data.loc[
                all_data["_row_id"].isin(ids),
                original_columns
            ]
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

    repaired = {
        "train": materialize(split_ids["train"]),
        "validation": materialize(split_ids["validation"]),
        "test": materialize(split_ids["test"]),
    }

    # Subset sizes and per-class counts must be unchanged
    before = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for name in before:
        old_counts = (
            before[name]["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        new_counts = (
            repaired[name]["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        if (
            len(before[name]) != len(repaired[name])
            or old_counts != new_counts
        ):
            raise RuntimeError(
                f"{name}: subset size or per-class counts changed during the repair"
            )

    return (
        repaired["train"],
        repaired["validation"],
        repaired["test"],
        initially_missing,
        swap_count,
    )


def main():
    train = load("training")
    val = load("validation")
    test = load("test")
    all_data = pd.concat([train, val, test], ignore_index=True).reset_index(drop=True)
    all_data["_row_id"] = all_data.index
    print(f"Source pool: {len(all_data)} pairs from {SRC}/")

    present = sorted(all_data["label"].unique())
    miss_src = sorted(set(range(NUM_CLASSES)) - set(present))
    if miss_src:
        print(f"[Warning] Classes absent from the source data (they cannot appear in any subset): {miss_src}")
    print(f"Source data contain {len(present)}/{NUM_CLASSES} classes\n")

    train_parts, val_parts, test_parts = [], [], []
    warn_no_val, warn_train_only = [], []

    for lab in present:
        rows = all_data[all_data["label"] == lab].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(rows)

        if n == 1:
            # A single pair goes to the test set; the class then has no training example.
            test_parts.append(rows)
            warn_train_only.append((int(lab), n))
            continue
        if n == 2:
            # Two pairs: one for training and one for testing, none for validation.
            train_parts.append(rows.iloc[[0]])
            test_parts.append(rows.iloc[[1]])
            warn_no_val.append((int(lab), n))
            continue

        # n >= 3: at least one test pair and one validation pair; the rest is used for training.
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        if n_test + n_val >= n:  # keep at least one training pair in very small classes
            n_test, n_val = 1, 1
        test_parts.append(rows.iloc[:n_test])
        val_parts.append(rows.iloc[n_test:n_test + n_val])
        train_parts.append(rows.iloc[n_test + n_val:])

    def cat(parts):
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df = cat(train_parts)
    val_df = cat(val_parts) if val_parts else pd.DataFrame(columns=all_data.columns)
    test_df = cat(test_parts)

    train_df, val_df, test_df, initially_missing, swap_count = repair_transductive(
        all_data,
        train_df,
        val_df,
        test_df,
    )

    print(
        f"Strict-transductive repair: {len(initially_missing)} drugs initially absent from training; "
        f"{swap_count} same-class swaps applied."
    )

    # Checks
    def cover(df):
        return set(df["label"].unique())

    print(f"train: {len(train_df):>8} pairs, {len(cover(train_df))} classes")
    print(f"val:   {len(val_df):>8} pairs, {len(cover(val_df))} classes")
    print(f"test:  {len(test_df):>8} pairs, {len(cover(test_df))} classes")

    miss_test = sorted(set(present) - cover(test_df))
    if miss_test:
        print(f"\nTest coverage: {len(cover(test_df))}/{len(present)} classes; missing: {miss_test}")
    else:
        print(f"\nThe test set covers all {len(present)} classes of the source data.")

    if warn_no_val:
        print(f"[Note] Classes with two pairs have no validation pair: {[l for l, _ in warn_no_val]}")
    if warn_train_only:
        print(f"[Warning] Classes with a single pair occur only in the test set and cannot be learned: "
              f"{[l for l, _ in warn_train_only]}")

    train_drugs = set(train_df["drug_1"]) | set(train_df["drug_2"])
    val_drugs = set(val_df["drug_1"]) | set(val_df["drug_2"])
    test_drugs = set(test_df["drug_1"]) | set(test_df["drug_2"])

    unseen_val = sorted(val_drugs - train_drugs)
    unseen_test = sorted(test_drugs - train_drugs)

    if unseen_val or unseen_test:
        raise RuntimeError(
            "Strict-transductive check failed: "
            f"unseen validation drugs={unseen_val}; "
            f"unseen test drugs={unseen_test}"
        )

    print("Strict-transductive check passed: every validation and test drug occurs in training.")

    # Save; the source files are left unchanged.
    out_dir = os.path.join(DATA_ROOT, OUT)
    os.makedirs(out_dir, exist_ok=True)
    # CRLF line endings make the written files byte-identical (same SHA-256) on every operating system.
    train_df.to_csv(
        os.path.join(out_dir, "ddi_training.csv"),
        index=False,
        lineterminator="\r\n",
    )
    val_df.to_csv(
        os.path.join(out_dir, "ddi_validation.csv"),
        index=False,
        lineterminator="\r\n",
    )
    test_df.to_csv(
        os.path.join(out_dir, "ddi_test.csv"),
        index=False,
        lineterminator="\r\n",
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

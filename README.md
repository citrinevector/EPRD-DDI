# EPRD-DDI

Source code for *EPRD-DDI: Explicit Pair Relations and Dynamic Prototype Regularization for Drug–Drug Interaction Event Prediction*.

EPRD-DDI predicts the interaction event (86 DrugBank event types) of an ordered drug pair. Each drug is represented by a 2D molecular image and a motif-token sequence. Reciprocal cross-attention fuses the two views, element-wise product and absolute-difference features pass explicit pair relations to the classifier, and a momentum-updated bank of event prototypes adds an auxiliary loss during training only; predictions come from the classifier alone.

## Requirements

Install the packages listed in `requirements.txt`. RDKit is needed only for the preprocessing scripts in `dataprocess/`.

## Data

The 86-event DrugBank benchmark is available from the [R2-DDI repository](https://github.com/linjc16/R2-DDI); no data are redistributed here. Its three files contain 191,808 records. Before the split is built, records that repeat an unordered drug pair are removed: the files are read in the order training, validation, test, (A, B) and (B, A) are treated as the same pair, and only the first record of each pair is kept. This leaves 191,402 records (1,706 drugs, event labels 0–85). Place the files as follows:

| Path | Content |
| --- | --- |
| `datasets/drugbank/0/clean_v1/ddi_training.csv` | deduplicated interaction records (`drug_1`, `drug_2`, `label`) |
| `datasets/drugbank/0/clean_v1/ddi_validation.csv` | deduplicated interaction records |
| `datasets/drugbank/0/clean_v1/ddi_test.csv` | deduplicated interaction records |
| `datasets/drugbank/drug_smiles.csv` | drug IDs and canonical SMILES (`drug_id`, `smiles`) |

## Preprocessing

Build and verify the strict-transductive split (split seed 42):

```bash
python datasets/build_transductive_86.py
python verify_transductive_split.py \
  --split-dir ./datasets/drugbank/0/transductive_strict \
  --num-classes 86 \
  --require-validation-all-classes \
  --check-builder-allocation
```

The split contains 133,979 training, 19,139 validation, and 38,284 test pairs. No drug pair occurs in more than one subset, and every validation and test drug occurs in training.

Generate the two molecular views:

```bash
python dataprocess/smiles2img.py --dataroot ./datasets/
python dataprocess/motif_utils.py
```

These commands create `datasets/drug_image/drugbank/`, `datasets/drug_image.csv`, `datasets/motif_vacb.csv`, and `datasets/drug_substructure_dict.npz`. Motif IDs start at 1; ID 0 is padding, and the vocabulary size plus 1 is the separator.

## Training and evaluation

```bash
SEED=2021
python -u train.py --runseed ${SEED} --log_dir ./logs/full_seed${SEED}
```

The default arguments reproduce the reported configuration:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--hidden_dim` | 512 | hidden width *D* |
| `--max_motif_len` | 8 | motif tokens retained per drug *L* |
| `--fusion_layers` | 1 | number of reciprocal-fusion layers |
| `--temperature` | 0.1 | prototype temperature τ |
| `--prototype_momentum` | 0.99 | prototype momentum μ |
| `--proto_weight` | 0.1 | prototype-loss weight λ |
| `--lr` | 1e-4 | peak learning rate (Adam) |
| `--weight_decay` | -6 | weight decay 10^-6 |
| `--warmup_epochs` / `--epochs` | 5 / 100 | linear warm-up followed by cosine annealing |
| `--batch` / `--grad_clip` | 128 / 1.0 | mini-batch size / gradient-norm clipping |
| `--patience` | 30 | early stopping on validation Macro-F1 |

The reported runs use seeds 2021, 2022, and 2023 on the same split; change only `--runseed`. Each run writes two files to `--log_dir`: `results_full.log`, with the loss, accuracy, Macro-F1, Macro-Precision, Macro-Recall, and Cohen's kappa of the training, validation, and test sets for every epoch (indexed from 0) and for the epoch with the highest validation Macro-F1; and `valid_best.pth`, the corresponding checkpoint. The ResNet-18 image encoder is randomly initialized.

## License

MIT License. Portions of the implementation are adapted from the ImageDDI codebase, released under the MIT License; the original copyright notice is retained in `LICENSE`.

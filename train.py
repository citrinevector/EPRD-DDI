"""Train and evaluate EPRD-DDI on the strict-transductive DrugBank split."""
import argparse
import logging
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms

from dataloader.ddi_dataloader import DDIDataset, get_datasets
from eprd_ddi import EPRD_DDI
from model.model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed
from utils.public_utils import cal_torch_model_params

warnings.filterwarnings("ignore")

SPLIT_DIRS = {'transductive': 'transductive_strict'}


def parse_args():
    parser = argparse.ArgumentParser(description="Train EPRD-DDI on the strict-transductive split.")
    # data
    parser.add_argument('--dataset', type=str, default="drugbank", help='dataset name')
    parser.add_argument('--dataroot', type=str, default="./datasets/", help='data root')
    parser.add_argument('--split_mode', type=str, default='transductive', choices=['transductive'],
                        help='reads <dataroot>/<dataset>/0/transductive_strict')
    parser.add_argument('--num_classes', type=int, default=86)
    parser.add_argument('--imageSize', type=int, default=224, help='input image size')
    parser.add_argument('--max_motif_len', type=int, default=8, help='motif tokens retained per drug (L)')
    parser.add_argument('--gpu', default='0', type=str, help='GPU index; -1 for CPU')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # model
    parser.add_argument('--image_model', type=str, default="ResNet18", choices=["ResNet18"], help='image backbone')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden width D')
    parser.add_argument('--fusion_layers', type=int, default=1, help='number of reciprocal-fusion layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='prototype temperature tau')
    parser.add_argument('--prototype_momentum', type=float, default=0.99, help='prototype momentum mu')
    parser.add_argument('--proto_weight', type=float, default=0.1, help='prototype-loss weight lambda')
    # optimization
    parser.add_argument('--lr', default=0.0001, type=float, help='peak learning rate')
    parser.add_argument('--weight_decay', default=-6, type=float, help='weight decay as a power of 10 (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='linear warm-up epochs')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient-norm clipping threshold')
    parser.add_argument('--patience', type=int, default=30,
                        help='stop after more than this many epochs without improvement in validation Macro-F1')
    parser.add_argument('--runseed', type=int, default=2021, help='training seed')
    # output
    parser.add_argument('--log_dir', default='./logs/eprd_ddi', help='directory for the log and checkpoint')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 saves the checkpoint with the highest validation Macro-F1')
    return parser.parse_args()


def get_file_path(dataroot, dataset, split_mode='transductive'):
    split_dir = os.path.join(dataroot, dataset, '0', SPLIT_DIRS[split_mode])
    assert os.path.isdir(split_dir), f"Split directory not found: {split_dir}"
    files = {
        'train': os.path.join(split_dir, 'ddi_training.csv'),
        'validation': os.path.join(split_dir, 'ddi_validation.csv'),
        'test': os.path.join(split_dir, 'ddi_test.csv'),
        'motif_vocab': os.path.join(dataroot, "motif_vacb.csv"),
    }
    for path in files.values():
        assert os.path.exists(path), f"File not found: {path}"
    print(f"[Split] {split_mode} -> {split_dir}")
    return files


def setup_logger(log_dir):
    """File logger for epoch-level metrics (epochs are indexed from 0)."""
    logger = logging.getLogger('eprd_ddi')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(os.path.join(log_dir, 'results_full.log'), mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    flog = setup_logger(args.log_dir)
    flog.info(f"split_mode={args.split_mode}  lr={args.lr}  batch={args.batch}  epochs={args.epochs}")
    flog.info("=" * 90)
    flog.info(f"{'Epoch':>5}  {'Split':>5}  {'Loss':>8}  {'Acc':>8}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'Kappa':>8}")
    flog.info("-" * 90)

    if args.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    image_folder, image_table = get_datasets(args.dataset, args.dataroot)
    files = get_file_path(args.dataroot, args.dataset, args.split_mode)

    if args.gpu == '-1':
        device, device_ids = torch.device('cpu'), []
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device_ids = [0] if torch.cuda.is_available() else []
    print(f"Device: {device}")

    # Seeds for Python, NumPy, PyTorch (CPU and CUDA) and the data-loader workers.
    fix_train_random_seed(args.runseed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.runseed)

    # Training images are augmented; validation and test images are only center-cropped.
    train_transform = transforms.Compose([
        transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=15),
        transforms.ToTensor()])
    eval_transform = transforms.Compose([transforms.CenterCrop(args.imageSize), transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def make_loader(split, transform, shuffle):
        dataset = DDIDataset(image_folder, image_table, files['motif_vocab'], files[split],
                             img_transformer=transform, normalize=normalize,
                             max_motif_len=args.max_motif_len)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True,
                                             worker_init_fn=seed_worker, generator=g)
        return dataset, loader

    train_dataset, train_dataloader = make_loader('train', train_transform, True)
    _, val_dataloader = make_loader('validation', eval_transform, False)
    _, test_dataloader = make_loader('test', eval_transform, False)
    print(f"[Preflight] max_motif_len={args.max_motif_len}, pair-sequence length={2 * args.max_motif_len + 1}")

    base_model = load_model(args.image_model)
    model = EPRD_DDI(
        num_classes=args.num_classes,
        vocab_size=train_dataset.vacb_size,
        hidden_dim=args.hidden_dim,
        max_motif_len=args.max_motif_len,
        cross_attn_layers=args.fusion_layers,
        temperature=args.temperature,
        prototype_momentum=args.prototype_momentum,
        proto_weight=args.proto_weight,
        dropout=args.dropout,
        base_model=base_model,
    )
    print(model)
    print("params: {}".format(cal_torch_model_params(model)))
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=10 ** args.weight_decay,
    )

    def lr_lambda(epoch):
        # Linear warm-up followed by cosine annealing.
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    eval_criterion = torch.nn.CrossEntropyLoss()

    results = None
    best_val_f1 = 0.0
    early_stop = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader,
                                  device=device, epoch=epoch, grad_clip=args.grad_clip)
        train_loss, train_results, _ = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                             criterion=eval_criterion, device=device,
                                                             epoch=epoch, return_data_dict=True)
        val_loss, val_results, _ = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                         criterion=eval_criterion, device=device,
                                                         epoch=epoch, return_data_dict=True)
        test_loss, test_results, _ = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                           criterion=eval_criterion, device=device,
                                                           epoch=epoch, return_data_dict=True)
        scheduler.step()

        for tag, loss_val, res in (('train', train_loss, train_results),
                                   ('val', val_loss, val_results),
                                   ('test', test_loss, test_results)):
            flog.info(f"{epoch:>5}  {tag:>5}  {loss_val:>8.4f}  {res['acc']:>8.4f}  {res['f1_macro']:>8.4f}  "
                      f"{res['precision']:>8.4f}  {res['recall']:>8.4f}  {res['kappa']:>8.4f}")
        print({'epoch': epoch, 'patience': early_stop, 'val_f1_macro': val_results['f1_macro']})

        # Checkpoint selection and early stopping use validation Macro-F1 only.
        if val_results['f1_macro'] > best_val_f1:
            best_val_f1 = val_results['f1_macro']
            results = {'highest_valid': val_results, 'final_train': train_results, 'final_test': test_results}
            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > args.patience:
                break

    flog.info("=" * 90)
    flog.info("BEST EPOCH (by val F1)")
    flog.info("-" * 90)
    for tag, res in (('train', results['final_train']), ('val', results['highest_valid']), ('test', results['final_test'])):
        flog.info(f"  {tag:>5}:  Acc={res['acc']:.4f}  F1={res['f1_macro']:.4f}  Prec={res['precision']:.4f}  "
                  f"Recall={res['recall']:.4f}  Kappa={res['kappa']:.4f}")
    flog.info("=" * 90)
    print("Test metrics at the best validation epoch:", results['final_test'])


if __name__ == "__main__":
    main(parse_args())

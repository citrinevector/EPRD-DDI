import logging
import os
import sys
import torch
import torchvision
from tqdm import tqdm
from model.evaluate import metric as utils_evaluate_metric


def get_support_model_names():
    return ["ResNet18"]


def load_model(modelname="ResNet18"):
    """Randomly initialized torchvision ResNet-18 (no pretrained weights)."""
    assert modelname in get_support_model_names()
    model = torchvision.models.resnet18(pretrained=False)
    return model


def train_one_epoch_multitask(model, optimizer, data_loader, device, epoch, grad_clip=1.0):
    """One training epoch with the full objective (cross-entropy + weighted prototype loss)."""
    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images_1, images_2, motif_seq, attention_mask, labels = data
        attention_mask = attention_mask.to(device)
        images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)
        motif_seq = motif_seq.to(device)

        pred, feature, info = model(
            images_1, images_2, motif_seq, attention_mask, labels, return_loss=True
        )
        loss = info['loss']

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification", return_data_dict=False):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    y_scores, y_true, y_pred, y_prob, features = [], [], [], [], []
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images_1, images_2, motif_seq, attention_mask, labels = data
        attention_mask = attention_mask.to(device)
        images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)
        motif_seq = motif_seq.to(device)
        with torch.no_grad():
            pred, feature, info = model(images_1, images_2, motif_seq, attention_mask)
            features.append(feature.cpu())
            if task_type == "classification":
                is_valid = labels != -1
                labels = labels.long()
                loss_mat = criterion(pred, labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        y_true.append(labels.cpu())
        y_pred.append(pred.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    features = torch.cat(features, dim=0).numpy()
    if return_data_dict:
        data_dict = {"y_true": y_true, "y_pred": y_pred, "features": features}
        return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, empty=-1), data_dict
    else:
        return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, empty=-1), features


def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None, logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
        "epoch": epoch,
        "model_state_dict": model_cpu,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler,
        "loss": loss,
        "result_dict": result_dict
    }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))
    filename = "{}/{}.pth".format(save_path, filename_pre)
    torch.save(state, filename)
    log.info("model has been saved as {}".format(filename))

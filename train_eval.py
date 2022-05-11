import math
import multiprocessing as mp
import os
import time

import matplotlib
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam, AdamW

# from torch_geometric.data import DataLoader ## Only use if using newer pyg version
from torch_geometric.data import DataLoader
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
from regularization.models import get_linear_schedule_with_warmup




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_multiple_epochs(
    train_dataset,
    test_dataset,
    model,
    epochs,
    batch_size,
    lr,
    lr_decay_factor,
    lr_decay_step_size,
    weight_decay,
    ARR=0,
    test_freq=1,
    logger=None,
    continue_from=None,
    res_dir=None,
    args=None,
):

    rmses = []

    if train_dataset.__class__.__name__ == "MyDynamicDataset":
        num_workers = mp.cpu_count()
    else:
        num_workers = 2


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    if test_dataset.__class__.__name__ == "MyDynamicDataset":
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    model.to(device).reset_parameters()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    if continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(res_dir, "model_checkpoint{}.pth".format(continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(res_dir, "optimizer_checkpoint{}.pth".format(continue_from)))
        )
        start_epoch = continue_from + 1
        epochs -= continue_from

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    batch_pbar = len(train_dataset) >= 100000
    pbar = range(start_epoch, epochs + start_epoch)
    lr_sched = get_linear_schedule_with_warmup(optimizer, 5, args.epochs)

    for epoch in pbar:
        train_loss = train(
            model,
            optimizer,
            train_loader,
            device,
            regression=True,
            ARR=ARR,
            show_progress=True,
            epoch=epoch,
            args=args,
        )

        if args.wandb:
            wandb.log({"epoch": epoch})
        if epoch % test_freq == 0:
            test_rmse = eval_rmse(model, test_loader, device, show_progress=batch_pbar)
            if args.wandb:
                wandb.log({"val_loss": test_rmse})

            rmses.append(test_rmse)
        else:
            rmses.append(np.nan)
        eval_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_rmse": rmses[-1],
        }
        # if not batch_pbar:
        #     pbar.set_description(
        #         'Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values())
        #     )
        print("Epoch {}, train loss {:.6f}, val rmse {:.6f}".format(*eval_info.values()))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_decay_factor * param_group["lr"]

        if logger is not None:
            logger(eval_info, model, optimizer, args.testing)

        lr_sched.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Final Test RMSE: {rmses[-1]:.6f}")

    return rmses[-1]


def test_once(test_dataset, model, batch_size, logger=None, ensemble=False, checkpoints=None):

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    
    if ensemble and checkpoints:
        rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
    else:
        rmse = eval_rmse(model, test_loader, device, show_progress=True)

    print(f"Test Once RMSE: {rmse:.6f}")
    epoch_info = "test_once" if not ensemble else "ensemble"
    eval_info = {
        "epoch": epoch_info,
        "train_loss": 0,
        "test_rmse": rmse,
    }
    if logger is not None:
        logger(eval_info, None, None)
    return rmse


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(
    model,
    optimizer,
    loader,
    device,
    regression=False,
    ARR=0,
    show_progress=False,
    epoch=None,
    args=None,
):
    model.train()
    total_loss = 0
    if show_progress:
        pbar = tqdm(loader)
    else:
        pbar = loader
    for ith, data in enumerate(pbar):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data, epoch)
        if regression:
            loss = F.mse_loss(out, data.y.view(-1))
        else:
            loss = F.nll_loss(out, data.y.view(-1))
        if show_progress:
            pbar.set_description("Epoch {}, batch loss: {}".format(epoch, loss.item()))

        if args.scenario in [1, 2, 3, 4, 5, 6, 7, 8]:
            for gconv in model.convs:
                w = torch.matmul(
                    gconv.att, 
                    gconv.basis.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
        elif args.scenario in [9, 10, 11, 12, 13, 14, 15, 16]:
            w = model.edge_embd.weight
            reg_loss = torch.sum((w[1:] - w[:-1]) ** 2)
        loss += ARR * reg_loss

        if args.wandb and ith % 5 == 0:
            wandb.log({"train_loss_step": loss})

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


def eval_loss(model, loader, device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print("Testing begins...")
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data, is_training=False)
        if regression:
            loss += F.mse_loss(out, data.y.view(-1), reduction="sum").item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction="sum").item()
        torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse(model, loader, device, show_progress=False):
    mse_loss = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


def eval_loss_ensemble(model, checkpoints, loader, device, regression=False, show_progress=False):
    loss = 0
    Outs = []
    for i, checkpoint in enumerate(checkpoints):
        if show_progress:
            print("Testing begins...")
            pbar = tqdm(loader)
        else:
            pbar = loader
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        outs = []
        if i == 0:
            ys = []
        for data in pbar:
            data = data.to(device)
            if i == 0:
                ys.append(data.y.view(-1))
            with torch.no_grad():
                out = model(data, is_training=False)
                outs.append(out)
        if i == 0:
            ys = torch.cat(ys, 0)
        outs = torch.cat(outs, 0).view(-1, 1)
        Outs.append(outs)
    Outs = torch.cat(Outs, 1).mean(1)
    if regression:
        loss += F.mse_loss(Outs, ys, reduction="sum").item()
    else:
        loss += F.nll_loss(Outs, ys, reduction="sum").item()
    torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse_ensemble(model, checkpoints, loader, device, show_progress=False):
    mse_loss = eval_loss_ensemble(model, checkpoints, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse


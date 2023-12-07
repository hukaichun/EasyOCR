import typing as tp
import logging

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm
from beautifultable import BeautifulTable

from model import FRCNNDetector

LOGGER = logging.getLogger(__name__)


def _set_target_to(targets, DEVICE:str):
    for target in targets:
        for k in target:
            target[k] = target[k].to(DEVICE)

def _print_result_wrapper(func):
    def wrapper(*args, **kwarg):
        result:tp.Dict = func(*args, **kwarg)
        table = BeautifulTable()
        
        keys = list(result.keys())
        row = [result[k].mean() for k in keys]
        table.columns.header = keys
        table.rows.append(row)
        LOGGER.info("[%s]\n%s", func.__name__, table)
        return result

    return wrapper



@_print_result_wrapper
def finetune_epoch(*training_set_loaders:torch.utils.data.DataLoader,
                   model: FRCNNDetector,
                   optimizer:torch.optim.Optimizer,
                   gradient_clip:float,
                   DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    losses_log = {}
    training_set_loaders = [training_set_loader for training_set_loader in training_set_loaders]
    np.random.shuffle(training_set_loaders)
    model.train()
    for training_set_loader in training_set_loaders:
        model = model.to(DEVICE)
        pbar = tqdm(training_set_loader)
        pbar.set_description("training phase")
        for images, targets in pbar:
            images = images.to(DEVICE)
            _set_target_to(targets, DEVICE)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            
            for k, v in loss_dict.items():
                losses_log.setdefault(k, []).append(v.detach().cpu().numpy())

    for k in losses_log:
        losses_log[k] = np.asarray(losses_log[k])

    return losses_log


@_print_result_wrapper
def verification(*verification_set_loaders:torch.utils.data.DataLoader,
                 model: FRCNNDetector,
                 DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    metric = MeanAveragePrecision()
    model = model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        for verification_set_loader in verification_set_loaders:
            pbar = tqdm(verification_set_loader)
            pbar.set_description("verification phase")
            for images, targets in pbar:
                images = images.to(DEVICE)
                predicts = model(images)
                _set_target_to(predicts, "cpu")
                metric.update(predicts, targets)
    
    losses_log = metric.compute()
    for k in losses_log:
        losses_log[k] = np.asarray(losses_log[k])
    return losses_log
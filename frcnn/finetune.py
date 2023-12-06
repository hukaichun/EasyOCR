import typing as tp
import logging

import torch
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
def finetune_epoch(model: FRCNNDetector,
                   optimizer:torch.optim.Optimizer,
                   training_set_loader:torch.utils.data.DataLoader,
                   gradient_clip:float,
                   *,
                   DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    losses = {}
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
            losses.setdefault(k, []).append(v.detach().cpu().numpy())
        
        break
    for k in losses:
        losses[k] = np.asarray(losses[k])
    return losses

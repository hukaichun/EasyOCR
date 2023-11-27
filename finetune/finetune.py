import typing as tp
import logging

from tqdm import tqdm
from beautifultable import BeautifulTable

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False

import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter


LOGGER = logging.getLogger(__name__)


def _print_result_wrapper(func):
    def wrapper(*args, **kwarg):
        result:tp.Dict = func(*args, **kwarg)
        table = BeautifulTable()
        table.columns.header = ["CTCLoss", "Accuracy", "Norm_ED"]
        table.rows.append([result.get("CTCLoss", "--"), result.get("Accuracy", "--"), result.get("Morm_ED", "--")])
        LOGGER.info("[%s]\n%s", func.__name__, table)
        return result

    return wrapper


def get_easyocr_recognizer_and_training_converter(lang_list:tp.List[str], model_ckpt:str=""):
    import easyocr
    def get_training_convertor(ref_converter:easyocr.utils.CTCLabelConverter):
        if isinstance(ref_converter, CTCLabelConverter):
            return ref_converter
        character = ''.join(ref_converter.character[1:])
        converter = CTCLabelConverter(character)
        converter.separator_list = ref_converter.separator_list
        converter.ignore_idx = ref_converter.ignore_idx
        converter.dict_list = ref_converter.dict_list
        converter.dict = ref_converter.dict
        return converter
    reader = easyocr.Reader(lang_list)
    recognizer = reader.recognizer
    if model_ckpt:
        ckpt = torch.load(model_ckpt)
        recognizer.load_state_dict(ckpt, strict=False)
    ref_converter = reader.converter
    training_converter = get_training_convertor(ref_converter)
    return recognizer, training_converter, reader

@_print_result_wrapper
def finetune_epoch(model:torch.nn.Module, 
                   criterion:torch.nn.CTCLoss, 
                   convertor:CTCLabelConverter, 
                   optimizer:torch.optim.Optimizer, 
                   training_set_loader:torch.utils.data.DataLoader, 
                   gradient_clip:float,
                   *,
                   DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    losses = []

    pbar = tqdm(training_set_loader)
    pbar.set_description("training phase")
    for image_tensors, labels in pbar:
        image = image_tensors.to(DEVICE)
        text, length = convertor.encode(labels)
        batch_size = image.size(0)

        preds = model(image, text).log_softmax(2)
        preds_size = torch.IntTensor([[preds.size(1)]*batch_size])
        preds = preds.permute(1,0,2)

        torch.backends.cudnn.enabled = False
        cost = criterion(preds, text.to(DEVICE), preds_size.to(DEVICE), length.to(DEVICE))
        torch.backends.cudnn.enabled = True

        optimizer.zero_grad(set_to_none=True)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        losses.append(cost.cpu().detach().numpy())

    ctc_losses = np.asarray(losses)

    result = {
        "CTCLoss": ctc_losses.mean(),
        "CTCLosses": ctc_losses
    }
    return result


def recognize(model:torch.nn.Module,
              converter:CTCLabelConverter, 
              validation_set_loader:torch.utils.data.DataLoader,
              *,
              DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    result = []
    with torch.no_grad():
        pbar = tqdm(validation_set_loader)
        pbar.set_description("prediction phase")
        for image_tensors, labels in pbar:
            image = image_tensors.to(DEVICE)
            text, length = converter.encode(labels)
            batch_size = image.size(0)

            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)]*batch_size)

            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)

            for gt, pred in zip(labels, preds_str):
                result.append({"gt": gt, "pred": pred})
    model.train()
    return result



@_print_result_wrapper
def validation(model:torch.nn.Module, 
               criterion:torch.nn.CTCLoss, 
               converter:CTCLabelConverter, 
               validation_set_loader:torch.utils.data.DataLoader,
               *,
               DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    n_correct = 0
    length_of_data = 0
    losses = []
    norm_EDs = []
    confidence_score_list = []



    model.eval()
    with torch.no_grad():
        pbar = tqdm(validation_set_loader)
        pbar.set_description("validation phase")
        for image_tensors, labels in pbar:
            image = image_tensors.to(DEVICE)
            text, length = converter.encode(labels)
            batch_size = image.size(0)

            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)]*batch_size)

            cost = criterion(preds.log_softmax(2).permute(1,0,2), text, preds_size, length)

            # decoding phase
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)

            # compute accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for gt,pred,pred_max_prob in zip(labels, preds_str, preds_max_prob):
                if pred == gt:
                    n_correct+=1
                
                if len(gt) == 0 or len(pred) ==0:
                    norm_ED = 0
                elif len(gt) > len(pred):
                    norm_ED = 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED = 1 - edit_distance(pred, gt) / len(pred)

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                confidence_score_list.append(confidence_score)
                norm_EDs.append(norm_ED)

            length_of_data+=batch_size
            losses.append(cost.cpu().detach().numpy())
    
    model.train()
    accuracy = n_correct / float(length_of_data) 

    result = {
        "CTCLoss": np.asarray(losses).mean(),
        "Accuracy": accuracy,
        "Norm_ED": np.asarray(norm_EDs).mean()
    }
    return result












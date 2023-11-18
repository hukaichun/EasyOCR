import os
import logging
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOGGER = logging.getLogger(__name__)


def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params








def train(model, converter, show_number = 2, amp=False):
    """ dataset preparation """


    train_dataset = Batch_Balanced_Dataset(opt)

    
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    opt.num_class = len(converter.character)

    
    model.train() 
    

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    loss_avg = Averager()

    # freeze some layers
    try:
        if freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]


    optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)


    """ start training """
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    while(True):
        # train part
        optimizer.zero_grad(set_to_none=True)
        

        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        # assert False
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        preds = model(image, text).log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
        torch.backends.cudnn.enabled = True
        
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
        optimizer.step()
        loss_avg.add(cost)

        # validation part
        # if (i % opt.valInterval == 0) and (i!=0):
        #     print('training time: ', time.time()-t1)
        #     t1=time.time()
        #     elapsed_time = time.time() - start_time
        #     # for log
        #     with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
        #         model.eval()
        #         with torch.no_grad():
        #             valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
        #             infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
        #         model.train()

        #         # training loss and validation loss
        #         loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        #         loss_avg.reset()

        #         current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

        #         # keep best accuracy model (on valid dataset)
        #         if current_accuracy > best_accuracy:
        #             best_accuracy = current_accuracy
        #             torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
        #         if current_norm_ED > best_norm_ED:
        #             best_norm_ED = current_norm_ED
        #             torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
        #         best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

        #         loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
        #         print(loss_model_log)
        #         log.write(loss_model_log + '\n')

        #         # show some predicted results
        #         dashed_line = '-' * 80
        #         head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        #         predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
        #         #show_number = min(show_number, len(labels))
                
        #         start = random.randint(0,len(labels) - show_number )    
        #         for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
        #             if 'Attn' in opt.Prediction:
        #                 gt = gt[:gt.find('[s]')]
        #                 pred = pred[:pred.find('[s]')]

        #             predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
        #         predicted_result_log += f'{dashed_line}'
        #         print(predicted_result_log)
        #         log.write(predicted_result_log + '\n')
        #         print('validation time: ', time.time()-t1)
        #         t1=time.time()
        # save model per 1e+4 iter.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1

import os
import datetime
import typing as tp
import yaml

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import dataset as DS
import finetune as FT

import logging
logging.basicConfig(level=logging.WARNING)

def print_config(the_dict:tp.Dict, prefix=""):
    for k, v in the_dict.items():
        if isinstance(v, dict):
            print(prefix, k, ":")
            print_config(v, prefix+'\t')
        else:
            print(prefix, k, ":", v)

def load_config(config_path:str):
    with open(config_path, 'r', encoding="utf8") as f:
        config = yaml.safe_load(f)
    return config

def get_training_model_and_converter_and_optimizer(skeleton, 
                                     freeze_FeatureExtraction:bool, 
                                     freeze_SequenceModeling:bool,
                                     model_ckpt:str,
                                     lr:float,
                                     rho:float,
                                     eps:float):
    model, converter = FT.get_easyocr_recognizer_and_training_converter(skeleton) # skeleton=["ch_tra"]
    # setup model
    if model_ckpt:
        ckpt = torch.load(model_ckpt)
        model.load_state_dict(ckpt, strict=False)
    if freeze_FeatureExtraction:
        for param in model.module.FeatureExtraction.parameters():
            param.requires_grad = False
    if freeze_SequenceModeling:
        for param in model.module.SequenceModeling.parameters():
            param.requires_grad = False

    filtered_parameters = [p for p in filter(lambda p:p.requires_grad, model.parameters())]
    optimizer = optim.Adadelta(filtered_parameters, lr=lr, rho=rho, eps=eps)
    return model, converter, optimizer

def compute_gradient_norm(model):
    filtered_parameters = [p for p in filter(lambda p:p.requires_grad, model.parameters())]
    grad_norms = [p.grad.data.norm(2) for p in filtered_parameters]
    return sum(grad_norms)**0.5

def get_save_to_path(prefix:str):
    current_datetime = datetime.datetime.now()
    current_time_string = current_datetime.strftime("%Y%m%d_%H%M%S")
    full_path_name = "/".join(["./saved_models", prefix, current_time_string])

    if not os.path.isdir(full_path_name):
        os.makedirs(full_path_name)

    return full_path_name


def main(config_path:str="./config_files/finetuning_config.yaml"):
    config = load_config(config_path)
    print_config(config)

    ds_log_level = logging.getLevelName(config["DS_log_level"])
    DS.LOGGER.setLevel(ds_log_level)
    ft_log_level = logging.getLevelName(config["FT_log_level"])
    FT.LOGGER.setLevel(ft_log_level)

    training_config = config["training_config"]
    model, converter, optimizer = get_training_model_and_converter_and_optimizer(
        skeleton = training_config["skeleton"],
        freeze_FeatureExtraction = training_config["freeze_FeatureExtraction"],
        freeze_SequenceModeling = training_config["freeze_SequenceModeling"],
        model_ckpt = training_config["model_ckpt"],
        lr = training_config["lr"],
        rho = training_config["rho"],
        eps = training_config["eps"]
    )

    DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(DEVICE)

    training_set_config = config["training_data_config"]
    character = ''.join(converter.character[1:])
    training_set_loader = DS.load_dataset(
        *training_set_config["train_data"],
        character=character,
        label_max_length=training_set_config["label_max_length"],
        imgH=training_set_config["imgH"],
        imgW=training_set_config["imgW"],
        keep_ratio_with_pad=training_set_config["keep_ratio_with_pad"],
        contrast_adjust=training_set_config["contrast_adjust"],
        batch_size=training_set_config["batch_size"],
        shuffle=training_set_config["shuffle"],
        num_workers=training_set_config["workers"],
        prefetch_factor=training_set_config["prefetch_factor"]
    )

    validation_data_config = config["validation_data_config"]
    validation_set_loader = DS.load_dataset(
        *validation_data_config["valid_data"],
        character=character,
        label_max_length=validation_data_config["label_max_length"],
        imgH=validation_data_config["imgH"],
        imgW=validation_data_config["imgW"],
        keep_ratio_with_pad=validation_data_config["keep_ratio_with_pad"],
        contrast_adjust=validation_data_config["contrast_adjust"],
        batch_size=validation_data_config["batch_size"],
        shuffle=validation_data_config["shuffle"],
        num_workers=validation_data_config["workers"],
        prefetch_factor=validation_data_config["prefetch_factor"]
    )

    
    # setup learning rate scheduler
    leraning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 
                                                                         factor=training_config["learning_rate_reduce_factor"], 
                                                                         patience=training_config["patience"],
                                                                         min_lr=training_config["minimum_learning_rate"])


    save_path = get_save_to_path(config["experiment_name"])
    writer = SummaryWriter(save_path)
    print(save_path)

    best_acc = -1
    best_norm_ED = -1

    # assert False
    training_epoch = training_config["training_episode"]
    for epoch in range(training_epoch):
        training_info = FT.finetune_epoch(model, criterion, converter, optimizer, training_set_loader, training_config["grad_clip"])
        leraning_rate_scheduler.step(training_info["CTCLoss"])
        writer.add_scalar("Training/CTCLoss", training_info["CTCLoss"], epoch)
        writer.add_scalar("Training/Learning_Rate", leraning_rate_scheduler._last_lr[-1], epoch)
        writer.add_scalar("Training/Gradient_norm", compute_gradient_norm(model), epoch)
        writer.add_histogram("Training/CTCLosses", training_info["CTCLosses"], epoch)
    
        # print(training_info)
        validation_info = FT.validation(model, criterion, converter, validation_set_loader)
        writer.add_scalar("Validation/CTCLoss", validation_info["CTCLoss"], epoch)
        writer.add_scalar("Validation/Accuracy", validation_info["Accuracy"], epoch)
        writer.add_scalar("Validation/norm_ED", validation_info["Norm_ED"], epoch)
        # print(validation_info)

        torch.save(model.state_dict(), f'{save_path}/epoch_{epoch+1}.pth')
        if validation_info["Accuracy"]>best_acc:
            best_acc = validation_info["Accuracy"]
            torch.save(model.state_dict(), f'{save_path}/best_acc.pth')
        
        if validation_info["Norm_ED"]>best_norm_ED:
            best_norm_ED = validation_info["Norm_ED"]
            torch.save(model.state_dict(), f'{save_path}/best_norm_ED.pth')


if __name__ == "__main__":
    configs = [
        "./config_files/finetuning_config.yaml",
    ]

    for config in configs:
        main(config)
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from yaml.loader import SafeLoader
from sklearn.model_selection import KFold
from datetime import datetime
from tqdm import tqdm
from time import time
from src import utils
import numpy as np
from torch import nn
import shutil
import random
import json
import torch
import yaml
import os

from src.models.tim_net import TimNet
from src.dataset import SER_Dataset
from src.models.light_ser_cnn import Light_SER
from src.models.cnn_tranformer import CNN_Transformer

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cpu" if not torch.cuda.is_available() else config["device"]
        # self.set_random_state(int(config["seed"]))
        
        with open(self.config["data_config"]) as f:
            self.data_config = yaml.load(f, Loader=SafeLoader)

        self.prepare_diretories_and_logger()
        self.cre_loss = torch.nn.CrossEntropyLoss()
        
        model = self.init_model()
        print(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"num params: {params}")

        n_split = int(self.config["k_fold"])
        self.k_folds = KFold(
            n_splits=n_split, shuffle=True, 
            random_state=int(config["seed"]))
        
        if config["mode"] == "train":
            print("Prepare data for training: ")
            self.inputs, self.labels = utils.prepare_data(
                general_config=config,
                data_config = self.data_config)
        elif config["mode"] == "infer":
            pass
    
    def init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

    def set_random_state(self, seed):
        print(f'set random_seed = {seed}')
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        
    def prepare_diretories_and_logger(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

        log_dir = f"{self.config['log_dir']}/{current_time}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"logging into {log_dir}")
            
        checkpoint_dir = self.config["checkpoint_dir"]
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            print(f'mkdir {checkpoint_dir}')
        
        self.writer = SummaryWriter(
            log_dir=log_dir
        )
        
    def init_model(self):
        with open(self.config["model_config"]) as f:
            model_config = yaml.load(f, Loader=SafeLoader)
        self.model_config = model_config
        
        if "light_ser_cnn" in self.config["model_config"]:
            model = Light_SER(self.model_config).to(self.device)
        elif "tim_net" in self.config["model_config"]:
            model = TimNet(n_label=len(self.data_config["label"].keys())).to(self.device)
        elif "cnn_transformer" in self.config["model_config"]:
            model = CNN_Transformer().to(self.device)
        
        model.apply(self.init_weight)
        return model
            
    def init_optimizer(self, model):
        optimizer = Adam(
            params=model.parameters(),
            betas=(self.config["beta1"], self.config["beta2"]),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]))
        
        return optimizer
    
    def prepare_dataloader(self, inputs, labels, mode="train"):
        dataset = SER_Dataset(
            inputs, labels, mode=mode)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config["num_worker"]),
            pin_memory=True,
            drop_last=False,
            shuffle=True)
        
        return dataloader
    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu")
        
        model_state_dict = state_dict["model_state_dict"]
        optim_state_dict = state_dict["optim_state_dict"]
        
        print(f'load checkpoint from {path}')        
        
        return {
            "model_state_dict":model_state_dict,
            "optim_state_dict":optim_state_dict
        }
    def train(self):
        print("########## Start Training #########")
        results = []
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(
                self.inputs, self.labels, 
                test_size=float(self.config["test_size"]), 
                shuffle=True, random_state=int(self.config["seed"])
            )
        print("Train: ", train_inputs.shape)
        print("Test: ", test_inputs.shape)
        test_dl = self.prepare_dataloader(test_inputs, test_labels, mode="test")
        
        kfold_tqdm = tqdm(self.k_folds.split(train_inputs, train_labels))
        kfold_tqdm.set_description("K_Fold Training: ")
        
        for _fold, (train_indices, valid_indices) in enumerate(kfold_tqdm):
            train_inputs, train_labels, valid_inputs, valid_labels = self.inputs[train_indices], self.labels[train_indices], self.inputs[valid_indices], self.labels[valid_indices]

            train_dl = self.prepare_dataloader(train_inputs, train_labels)
            valid_dl = self.prepare_dataloader(valid_inputs, valid_labels, mode="test")
            
            print("################# init model ##################")
            model = self.init_model()
            print("############### init optimizer #################")
            optimizer = self.init_optimizer(model)
            
            model.train()
            best_acc, best_wa, best_uwa = -1, -1, -1
            for epoch in range(int(self.config["n_epoch"])):
                train_losses, valid_losses = [], []
                
                for i, batch in enumerate(train_dl):
                    optimizer.zero_grad()
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    preds = model(inputs=inputs, lengths=None)
                    
                    loss = self.cre_loss(preds, labels)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    loss.backward()
                    
                    train_losses.append(loss.item())
                    optimizer.step()

                if (epoch+1) % int(self.config["evaluate_per_epoch"])==0:
                    train_loss = np.mean(train_losses)
                    target_names = list(self.data_config["label"].keys())
                    # kfold_tqdm.write(f'# training loss at {epoch} = {train_loss}')
                                    
                    valid_results = self.evaluate(valid_dl=valid_dl, model=model)
                    valid_cls_result = classification_report(
                        y_pred=valid_results["predicts"], 
                        y_true=valid_results["labels"],
                        output_dict=True, zero_division=0,
                        target_names=target_names)       
                    
                    if best_acc < valid_cls_result["accuracy"]:
                        best_acc = valid_cls_result["accuracy"]
                        path = f'{self.config["checkpoint_dir"]}/best_acc_checkpoint.pt'
                        self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                        
                        # kfold_tqdm.write(f'# best accuracy at {epoch} = {best_acc}')
                        
                    if best_wa < valid_cls_result["weighted avg"]["recall"]:
                        best_wa = valid_cls_result["weighted avg"]["f1-score"]
                        path = f'{self.config["checkpoint_dir"]}/best_war_checkpoint.pt'
                        self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                        
                        # kfold_tqdm.write(f'# best weighted avg f1-score at {epoch} = {best_wa}')
                        
                    if best_uwa < valid_cls_result["macro avg"]["f1-score"]:
                        best_uwa = valid_cls_result["macro avg"]["f1-score"]
                        path = f'{self.config["checkpoint_dir"]}/best_uwar_checkpoint.pt'
                        self.save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, loss=train_loss)
                        
                        # kfold_tqdm.write(f'# best macro avg f1-score at {epoch} = {best_uwa}')
                    
                    # kfold_tqdm.write("############################################")
                    kfold_tqdm.set_postfix(
                        {
                            "macro_avg":best_uwa,
                            "weighted_avg":best_wa,
                            "accuracy":best_acc,
                            "epoch":epoch,
                            "valid_loss":valid_results["loss"],
                            "train_loss":train_loss
                            }
                        )  
            path = f'{self.config["checkpoint_dir"]}/best_acc_checkpoint.pt'
            test_results = self.test(checkpoint=path,test_dl=test_dl)
            os.remove(path)
            
            path = f'{self.config["checkpoint_dir"]}/best_war_checkpoint.pt'
            test_results = self.test(checkpoint=path,test_dl=test_dl)
            os.remove(path)
            
            path = f'{self.config["checkpoint_dir"]}/best_uwar_checkpoint.pt'
            test_results = self.test(checkpoint=path,test_dl=test_dl)
            os.remove(path)
            results.append([_fold, test_results["acc"], test_results["war"], test_results["uwar"]])
            
            print(results[-1])
            print(
                {
                    "macro_avg":best_uwa,
                    "weighted_avg":best_wa,
                    "accuracy":best_acc,
                    "epoch":epoch,
                    "valid_loss":valid_results["loss"],
                    "train_loss":train_loss
                    }
                )
        with open(f'{self.config["model_config"]}.txt', "w", encoding="utf-8") as f:
            np_results = np.array(results).mean(axis=0).tolist()
            results.append(np_results)
            
            print(results)
            for fold in results:
                fold = list(map(str, fold))
                f.write("\t".join(fold) + "\n")
                
    def save_checkpoint(self, path, model, optimizer, epoch, loss):
        state_dict = {
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss":loss,
            "epoch":epoch,
        }
        torch.save(state_dict, path)

    def evaluate(self, model, valid_dl, mode="test"):
        predicts, labels = [], []
        
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(valid_dl):
                inputs, _labels = batch
                inputs = inputs.to(self.device)
                _labels = _labels.to(self.device)
                
                preds = model(inputs=inputs, lengths=None)
                loss = self.cre_loss(preds, _labels)
                
                preds = torch.nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=-1)
                _labels = _labels.argmax(dim=-1)
                
                labels += _labels.cpu().tolist()
                predicts += preds.cpu().tolist()
                
                losses.append(loss.item())        
        return {
            "loss":torch.tensor(losses).mean(),
            "predicts":np.array(predicts),
            "labels":np.array(labels),
        }
    def test(self, checkpoint, test_dl):
        emo_predicts, emo_labels = [], []
        
        model = self.init_model()            
        state_dict = self.load_checkpoint(checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        target_names = list(self.data_config["label"].keys())
        with torch.no_grad():
            test_results = self.evaluate(valid_dl=test_dl, model=model)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=False, zero_division=0,
            target_names=target_names)
        
        print(test_cls_result)
        
        test_cls_result = classification_report(
            y_pred=test_results["predicts"], 
            y_true=test_results["labels"],
            output_dict=True, zero_division=0,
            target_names=target_names)
                
        return {
            "acc":test_cls_result["accuracy"],
            "war":test_cls_result["weighted avg"]["recall"],
            "uwar":test_cls_result["macro avg"]["recall"],
        }
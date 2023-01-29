# Adapted from https://github.com/Xylambda/blog/blob/master/_notebooks/2021-01-04-pytorch_trainer.ipynb

import os
import json
import time
import warnings
import torch

class Trainer:
    """Trainer
    
    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    attribute_index_path :
        Path to dict with attributes as keys and indices as values.
    type_index_path :
        Path to dict with types as keys and indices as values.
    attribute_types_full_path :
        Path to dict with attributes as keys and types as values.

    """
    def __init__(
        self, 
        model, 
        criterion, 
        optimizer,
        device=None,
        attribute_index_path: os.path = "data/attribute_index.json",
        type_index_path: os.path = "data/type_index.json",
        attribute_types_full_path: os.path = "data/attribute_types_full.json"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = self._get_device(device)
    
        with open(attribute_index_path) as f:
            attribute_index = json.load(f)
        with open(type_index_path) as f:
            type_index = json.load(f)
        with open(attribute_types_full_path) as f:
            attribute_types_full = json.load(f)

        # dict with key = type_id : value = list of attrib_ids 
        type2attrib = {i : [] for i in range(8)}
        for attribute in attribute_types_full:
            type2attrib[type_index[attribute_types_full[attribute]]].append(attribute_index[attribute])
        self.type2attrib = type2attrib
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []
        self.train_mean_acc_ = []
        self.val_mean_acc_ = []
        
    def fit(self, train_loader, val_loader, epochs):
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # train
            train_loss, train_mean_acc = self._train(train_loader)
            
            # validate
            val_loss, val_mean_acc =self._validate(val_loader)
            
            print('[{:d}/{:d}]\t loss/train: {:.5f}\t loss/val: {:.5f}\t mA/train: {:.2f}\t mA/val: {:.2f}'.format(
                epoch+1, epochs, train_loss, val_loss, train_mean_acc, val_mean_acc)
                )
            
            self.train_loss_.append(train_loss)
            self.val_loss_.append(val_loss)
            self.train_mean_acc_.append(train_mean_acc)
            self.val_mean_acc_.append(val_mean_acc)

            epoch_time = time.time() - epoch_start_time

        total_time = time.time() - total_start_time
        print(f"Training completed in {round(total_time, 5)} seconds")
    
    def _train(self, loader):
        self.model.train()

        cumulative_loss = 0.
        samples = 0.
        cumulative_acc_by_class = {i : 0. for i in range(8)}
        samples_by_class = {i : 0. for i in range(8)}
        
        for inputs, targets in loader:
            # move to device
            inputs, targets = self._to_device(inputs, targets, self.device)
            
            # forward pass
            out = self.model(inputs)
            
            # loss
            loss = self._compute_loss(out, targets)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()

            # n samples and cumulative accuracy by class 
            for type in range(8):
                predicted = torch.gather(input=out, dim=1, index=torch.LongTensor([self.type2attrib[type]]*inputs.shape[0])).round() # prediction round to 1 (positive) or 0 (negative, missing)
                samples_by_class[type] += len(self.type2attrib[type])*inputs.shape[0]
                cumulative_acc_by_class[type] += predicted.eq(torch.gather(input=targets, dim=1, index=torch.LongTensor([self.type2attrib[type]]*inputs.shape[0])).round()).sum().item()

            # n samples and cumulative loss
            samples += inputs.shape[0]
            cumulative_loss += loss.item()

            # train loss and train mean accuracy across classes
            train_loss = cumulative_loss/samples
            train_mean_accuracy = torch.mean(torch.tensor(list(cumulative_acc_by_class.values())) / torch.tensor(list(samples_by_class.values()))*100).item()

        return train_loss, train_mean_accuracy
    
    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)
    
    def _validate(self, loader):
        self.model.eval()

        cumulative_loss = 0.
        samples = 0.
        cumulative_acc_by_class = {i : 0. for i in range(8)}
        samples_by_class = {i : 0. for i in range(8)}
        
        with torch.no_grad():
            for inputs, targets in loader:
                # move to device
                inputs, targets = self._to_device(
                    inputs, 
                    targets, 
                    self.device
                )
                
                out = self.model(inputs)
                loss = self._compute_loss(out, targets)

                # n samples and cumulative accuracy by class
                for type in range(8):
                    predicted = torch.gather(input=out, dim=1, index=torch.LongTensor([self.type2attrib[type]]*inputs.shape[0])).round() # prediction round to 1 (positive) or 0 (negative, missing)
                    samples_by_class[type] += len(self.type2attrib[type])*inputs.shape[0]
                    cumulative_acc_by_class[type] += predicted.eq(torch.gather(input=targets, dim=1, index=torch.LongTensor([self.type2attrib[type]]*inputs.shape[0])).round()).sum().item()

                # n samples and cumulative loss
                samples += inputs.shape[0]
                cumulative_loss += loss.item()
                
                # val loss and val mean accuracy across classes
                val_loss = cumulative_loss/samples
                val_mean_accuracy = torch.mean(torch.tensor(list(cumulative_acc_by_class.values())) / torch.tensor(list(samples_by_class.values()))*100).item()

        return val_loss, val_mean_accuracy
    
    def _compute_loss(self, real, target):
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        # loss += penalty.item()
            
        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev
# Adapted from https://github.com/Xylambda/blog/blob/master/_notebooks/2021-01-04-pytorch_trainer.ipynb

import os
import json
import time
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from evaluator import Evaluator

class Trainer:
    """Trainer
    
    Parameters
    ----------
    model : torch.Module
        The model to train.
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
        optimizer,
        scheduler,
        model_type,
        config,
        glove_embeddings,
        loss_weights,
        loss_pos_weights,
        loss_neg_weights,
        device=None,
        attribute_index_path: os.path = "data/attribute_index.json",
        type_index_path: os.path = "data/type_index.json",
        attribute_types_full_path: os.path = "data/attribute_types_full.json"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type
        self.config = config
        self.glove_embeddings = glove_embeddings
        self.loss_weights = torch.load(loss_weights).to(device)
        self.loss_pos_weights = torch.load(loss_pos_weights).to(device)
        self.loss_neg_weights = torch.load(loss_neg_weights).to(device)
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
        # self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []
        self.train_mean_acc_ = []
        self.val_mean_acc_ = []
        self.val_map_ = []
        
    def fit(self, train_loader, val_loader, epochs):
        # track total training time
        total_start_time = time.time()

        best_val_map = 0

        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            self.model.train(True)
            # train
            train_loss, train_mean_acc = self._train(train_loader)

            self.model.train(False)
            # validate
            val_loss, val_mean_acc, val_map = self._validate(val_loader)
            # apply LR decay of 0.1 when val mAP doesn't improve for 2 epochs
            self.scheduler.step(val_map)

            # save best
            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(self.model.state_dict(), f"models/{self.model_type}_model.pt")
                print(f"New best model! Saved with mAP {best_val_map}")
            
            print('[{:d}/{:d}]\t loss/train: {:.5f}\t \
                loss/val: {:.5f}\t mA/train: {:.2f}\t \
                mA/val: {:.2f}\t mAP/val: {:.2f}'.format(
                    epoch+1, epochs, train_loss, val_loss,
                    train_mean_acc, val_mean_acc, val_map
                ))
            
            self.train_loss_.append(train_loss)
            self.val_loss_.append(val_loss)
            self.train_mean_acc_.append(train_mean_acc)
            self.val_mean_acc_.append(val_mean_acc)
            self.val_map_.append(val_map)

            epoch_time = time.time() - epoch_start_time

        total_time = time.time() - total_start_time
        print(f"Training completed in {round(total_time, 5)} seconds")
    
    def _train(self, loader):
        self.model.train()

        cumulative_loss = 0.
        samples = 0.
        cumulative_acc_by_class = {i : 0. for i in range(8)}
        samples_by_class = {i : 0. for i in range(8)}
        
        for batch in tqdm(loader):
            # move to device
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            inputs = batch["image"]
            targets = batch["attributes_label"]
            
            object_embeddings = self.glove_embeddings(batch["object_name"]).to(self.device)
            if self.model_type == "kd":
                teacher_logits, student_logits = self.model(inputs,
                                                            object_embeddings)
                loss_student = self._compute_loss(student_logits, targets)
                loss_distill = self._compute_kd_loss(
                    student_logits, teacher_logits, "xent")
                loss = loss_student + loss_distill
            else:
                # forward pass
                out = self.model(inputs, object_embeddings)
                # loss
                loss = self._compute_loss(out, targets)
            
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()

#            # n samples and cumulative accuracy by class 
#            for typ in range(8):
#                predicted = torch.gather(
#                    input=torch.sigmoid(out), 
#                    dim=1, 
#                    index=torch.LongTensor(
#                        [self.type2attrib[typ]]*inputs.shape[0]
#                    ).to(self.device)
#                ).round() # round to 1 (positive) or 0 (negative, missing)
#                samples_by_class[typ] += len(self.type2attrib[typ])*inputs.shape[0]
#                cumulative_acc_by_class[typ] += predicted.eq(
#                    torch.gather(
#                        input=targets, 
#                        dim=1, 
#                        index=torch.LongTensor(
#                            [self.type2attrib[typ]]*inputs.shape[0]
#                        ).to(self.device)
#                    ).round()
#                ).sum().item()

            # n samples and cumulative loss
            samples += inputs.shape[0]
            cumulative_loss += loss.item()

        # train loss and train mean accuracy across classes
        train_loss = cumulative_loss / samples
        train_mean_accuracy = torch.mean(
            torch.tensor(list(cumulative_acc_by_class.values())) \
          / torch.tensor(list(samples_by_class.values()))*100).item()

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
            full_predictions = []
            full_targets = []
            for batch in loader:
                # move to device
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                inputs = batch["image"]
                targets = batch["attributes_label"]

                object_embeddings = self.glove_embeddings(batch["object_name"]).to(self.device)
                if self.model_type == "kd":
                    teacher_logits, student_logits = self.model(inputs, object_embeddings)
                    loss_student = self._compute_loss(student_logits, targets)
                    loss_distill = self._compute_kd_loss(
                        student_logits, teacher_logits, "xent")
                    loss = loss_student + loss_distill
                    sig_out = torch.sigmoid(student_logits)
                else:
                    out = self.model(inputs, object_embeddings)
                    loss = self._compute_loss(out, targets)
                    sig_out = torch.sigmoid(out)

                full_predictions.append(sig_out)
                full_targets.append(targets)


                # n samples and cumulative accuracy by class
#                for typ in range(8):
#                    predicted = torch.gather(
#                        input=sig_out, 
#                        dim=1, 
#                        index=torch.LongTensor(
#                            [self.type2attrib[typ]]*inputs.shape[0]
#                        ).to(self.device)
#                    ).round() # round to 1 (positive) or 0 (negative, missing)
#                    samples_by_class[typ] += len(self.type2attrib[typ])*inputs.shape[0]
#                    cumulative_acc_by_class[typ] += predicted.eq(
#                        torch.gather(
#                            input=targets, 
#                            dim=1, 
#                            index=torch.LongTensor(
#                                [self.type2attrib[typ]]*inputs.shape[0]
#                            ).to(self.device)
#                        ).round()
#                    ).sum().item()

                # n samples and cumulative loss
                samples += inputs.shape[0]
                cumulative_loss += loss.item()

        # vaw_dataset evaluator
        full_targets = torch.concat(full_targets, dim=0)
        full_targets = torch.where(full_targets==-1, 2, full_targets)
        self.evaluator = Evaluator(
            fpath_attr2idx=self.config["attrib_idx_file"],
            fpath_attr_type=self.config["attrib_types"],
            fpath_attr_parent_type=self.config["attrib_parent_types"],
            fpath_attr_headtail=self.config["attrib_head_tail"]
        )

        scores_overall, scores_per_class = self.evaluator.evaluate(
            pred=torch.concat(full_predictions, dim=0),
            gt_label=full_targets,
            threshold_type="topk"
        )
        val_map = np.mean(
            [att.ap for att in 
             map(self.evaluator.get_score_class, range(self.evaluator.n_class))
            ])

        # val loss and val mean accuracy across classes
        val_loss = cumulative_loss / samples
        val_mean_accuracy = torch.mean(torch.tensor(list(cumulative_acc_by_class.values())) / torch.tensor(list(samples_by_class.values()))*100).item()

        return val_loss, val_mean_accuracy, val_map

    def test(self, loader):
        print("Running test inference with best validation model")
        self.model.load_state_dict(torch.load(f"models/{self.model_type}_model.pt"))
        self.model.eval()

        with torch.no_grad():
            full_predictions = []
            full_targets = []
            for batch in loader:
                # move to device
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                inputs = batch["image"]
                targets = batch["attributes_label"]

                object_embeddings = self.glove_embeddings(batch["object_name"]).to(self.device)
                if self.model_type == "kd":
                    teacher_logits, student_logits = self.model(inputs, object_embeddings)
                    sig_out = torch.sigmoid(student_logits)
                else:
                    out = self.model(inputs, object_embeddings)
                    sig_out = torch.sigmoid(out)

                full_predictions.append(sig_out)
                full_targets.append(targets)

        # vaw_dataset evaluator
        full_targets = torch.concat(full_targets, dim=0)
        full_targets = torch.where(full_targets==-1, 2, full_targets)
        self.evaluator = Evaluator(
            fpath_attr2idx=self.config["attrib_idx_file"],
            fpath_attr_type=self.config["attrib_types"],
            fpath_attr_parent_type=self.config["attrib_parent_types"],
            fpath_attr_headtail=self.config["attrib_head_tail"]
        )

        scores_overall, scores_per_class = self.evaluator.evaluate(
            pred=torch.concat(full_predictions, dim=0),
            gt_label=full_targets,
            threshold_type="topk"
        )
        test_map = np.mean(
            [att.ap for att in 
             map(self.evaluator.get_score_class, range(self.evaluator.n_class))
            ])

        torch.save(torch.concat(full_predictions, dim=0), f"eval/{self.model_type}_test_preds_{test_map}.pt")
        torch.save(full_targets, f"eval/{self.model_type}_test_gt_{test_map}.pt")
        print(f"Saved test predictions and GTs to eval/test_(preds/gt)_{test_map}.pt")

        return test_map

    def _compute_loss(self, real, target):
        # get indices of unlabeled attributes
        weights = torch.where(target==-1., 0.05*self.loss_weights, 1.*self.loss_weights)
        # missing labels have small target probability assigned with soft labels, 0 otherwise
        if self.model_type == "soft_labels":
            target = torch.where(target==-1., 0.05, target)
        else:
            target = torch.where(target==-1., 0., target)

        # have to do this manually to also include negative weights
        max_val = (-real).clamp_min(0)
        log_pos_weight = self.loss_pos_weights.mul(target)
        log_neg_weight = self.loss_neg_weights.mul(1 - target)
        log_weight = log_pos_weight + log_neg_weight
        # this kind of works. not sure about stability though, maybe max_val required
        loss = (1 - target).mul(real).add(((-real).exp().add(1)).log()).mul(log_weight)

        loss = loss * self.loss_weights
        loss = loss.mean()
            
        return loss

    def _compute_kd_loss(self, student, teacher, loss_type):
        if loss_type == "kl_div":
            log_student = F.log_softmax(student, dim=-1)
            log_teacher = F.log_softmax(teacher, dim=-1)
            loss = F.kl_div(log_student, log_teacher, reduction="batchmean", log_target=True)
        else: # cross entropy
            softmax_teacher = F.softmax(teacher, dim=-1)
            loss = F.cross_entropy(student, softmax_teacher)

        return loss


    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

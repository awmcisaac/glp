from dataset import VAWDataset, VAWSoftLabelDataset
from trainer import Trainer
from model import Model
from glove import GloVeEmbedding
from evaluator import Evaluator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json

def main(args, config):
    torch.manual_seed(42)
    dataset = VAWSoftLabelDataset if args.model_type == "soft_labels" else VAWDataset
    train_dataset = dataset(
        annotations_file=config["train_annotations"],
        attrib_idx_file=config["attrib_idx_file"],
        attrib_parent_types=config["attrib_parent_types"],
        attrib_types=config["attrib_types"],
        attrib_weights=config["attrib_weights"],
        img_dir=config["img_dir"],
        split="train",
        resize=224
    )
    val_dataset = dataset(
        annotations_file=config["val_annotations"],
        attrib_idx_file=config["attrib_idx_file"],
        attrib_parent_types=config["attrib_parent_types"],
        attrib_types=config["attrib_types"],
        attrib_weights=config["attrib_weights"],
        img_dir=config["img_dir"],
        split="val",
        resize=224
    )
    test_dataset = dataset(
        annotations_file=config["test_annotations"],
        attrib_idx_file=config["attrib_idx_file"],
        attrib_parent_types=config["attrib_parent_types"],
        attrib_types=config["attrib_types"],
        attrib_weights=config["attrib_weights"],
        img_dir=config["img_dir"],
        split="test",
        resize=224
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
#        sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=100),
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
#        sampler=torch.utils.data.RandomSampler(val_dataset, num_samples=100),
        num_workers=12,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
#        sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=100),
        num_workers=12,
        pin_memory=True
    )

    loss_weights = torch.load(config["loss_weights"]).to(args.device)
    loss_pos_weights = torch.load(config["loss_pos_weights"]).to(args.device)
    loss_neg_weights = torch.load(config["loss_neg_weights"]).to(args.device)

    model = Model(args.device)
    optimizer = torch.optim.Adam([
        {"params": model.model_resnet.parameters(), "lr": 1e-5},
        {"params": model.f_gate.parameters()},
        {"params": model.classifier.parameters()},
        {"params": model.adaptive_avg_pool.parameters()}
                 ], lr=7e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2
    )
    glove_embeddings = GloVeEmbedding(config["glove_embedding_file"])

    evaluator = Evaluator(
        fpath_attr2idx=config["attrib_idx_file"],
        fpath_attr_type=config["attrib_types"],
        fpath_attr_parent_type=config["attrib_parent_types"],
        fpath_attr_headtail=config["attrib_head_tail"]
    )

    trainer = Trainer(
        model, 
        optimizer, 
        scheduler,
        args.model_type,
        config,
        glove_embeddings=glove_embeddings,
        loss_weights=config["loss_weights"],
        loss_pos_weights=config["loss_pos_weights"],
        loss_neg_weights=config["loss_neg_weights"],
        device=args.device,
        attribute_index_path=config["attrib_idx_file"],
        type_index_path=config["type_idx_file"],
        attribute_types_full_path=config["attrib_types_full"]
    )

    trainer.fit(train_loader, val_loader, args.epochs)

    trainer.test(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/base.json")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="default",
        choices=["default", "soft_labels", "kd"])
    
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    main(args, config)


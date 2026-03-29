#!/usr/bin/env python3
import os
import sys
import argparse
import random
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

from transformers import CLIPProcessor, CLIPModel

# Add local paths for augmentation and similarity utilities.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_model_root = os.path.join(os.path.dirname(os.path.dirname(_this_dir)), "model")
sys.path.append(_model_root)
from augmentations import create_augmentation

sys.path.append("/Data3/Daniel")
from similarity_metrics import SimilarityMetricsCalculator

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "vandy": {
        "data_root": "/Data3/Daniel/Data1/Glom_Patches_0207_train_val_test_vandy",
        "class_prompts": {
            "Normal glomeruli": "A pathology image of a normal glomeruli",
            "Obsolescent glomeruli": "A pathology image of an obsolescent glomeruli",
            "Solidified glomeruli": "A pathology image of a solidified glomeruli",
            "Disappearing glomeruli": "A pathology image of a disappearing glomeruli",
            "Non-glomerular": "A pathology image of non-glomerular tissue",
        },
    },
    "cornell": {
        "data_root": "/Data3/Daniel/Data1/Glom_Patches_0207_train_val_test_cornell",
        "class_prompts": {
            "Atubular Glomeruli": "A pathology image of atubular glomerulus",
            "Global Glomerulosclerosis": "A pathology image of globally sclerotic glomerulus",
            "Ischemic Glomeruli": "A pathology image of ischemic glomerulus",
            "Segmental Glomerulosclerosis": "A pathology image of segmentally sclerotic glomerulus",
            "Viable Glomeruli": "A pathology image of viable glomerulus",
        },
    },
}

class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, processor, transform=None, class_prompts=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.transform = transform
        self.class_prompts = class_prompts or {}

        # Build the class index mapping from the CSV file.
        unique_classes = sorted(self.df['class_name'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.class_names = unique_classes

        # Validate that every class has a prompt entry.
        missing = set(unique_classes) - set(self.class_prompts.keys())
        if missing:
            raise ValueError(f"Missing classes in class_prompts: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        class_name = row['class_name']
        label = self.class_to_idx[class_name]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        text_prompt = self.class_prompts[class_name]

        image_inputs = self.processor(images=image, return_tensors="pt")
        text_inputs = self.processor(text=[text_prompt], return_tensors="pt", padding=True, truncation=True, max_length=77)

        return {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
        }

class CLIPContrastiveDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = {}
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        batch["pixel_values"] = pixel_values

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_length = max(ids.size(0) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - ids.size(0)
            if padding_length > 0:
                padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)], dim=0)
                padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)], dim=0)
            else:
                padded_ids = ids
                padded_mask = mask
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(padded_attention_mask)
        batch["labels"] = torch.stack([f["labels"] for f in features])
        batch["image_path"] = [f["image_path"] for f in features]
        return batch

def apply_layer_freezing(model, text_unfreeze_layers=0, vision_unfreeze_layers=0):
    """Freeze the full model and unfreeze the requested top layers."""
    for param in model.parameters():
        param.requires_grad = False

    model.logit_scale.requires_grad_(True)

    if text_unfreeze_layers > 0:
        total_text_layers = len(model.text_model.encoder.layers)
        start_idx = max(0, total_text_layers - text_unfreeze_layers)
        for i in range(start_idx, total_text_layers):
            for param in model.text_model.encoder.layers[i].parameters():
                param.requires_grad = True

    if vision_unfreeze_layers > 0:
        total_vision_layers = len(model.vision_model.encoder.layers)
        start_idx = max(0, total_vision_layers - vision_unfreeze_layers)
        for i in range(start_idx, total_vision_layers):
            for param in model.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True

    return model

def get_class_text_features(model, processor, class_names, device, class_prompts):
    """Encode class prompts into normalized text features."""
    with torch.no_grad():
        text_inputs = processor(
            text=[class_prompts[c] for c in class_names],
            return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        class_text_features = model.get_text_features(**text_inputs)
        class_text_features = F.normalize(class_text_features, dim=-1)
    return class_text_features

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, metric, model):
        if self.best_metric is None:
            self.best_metric = metric
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.restore_best_weights and self.best_weights:
            model.load_state_dict(self.best_weights)

def clip_contrastive_loss(image_embeds, text_embeds, logit_scale):
    """Compute the standard CLIP contrastive loss."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
    logits_per_text = logit_scale * (text_embeds @ image_embeds.T)
    batch_size = image_embeds.size(0)
    labels = torch.arange(batch_size, device=image_embeds.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2, logits_per_image, logits_per_text

def evaluate_with_class_prototypes(model, dataloader, device, class_names, class_prompts):
    """Evaluate using class-prompt prototypes."""
    model.eval()
    all_image_features = []
    all_labels = []

    processor = dataloader.dataset.processor
    class_text_features = get_class_text_features(model, processor, class_names, device, class_prompts)

    for batch in dataloader:
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

            all_image_features.append(image_features)
            all_labels.append(labels)

    image_features = torch.cat(all_image_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    pure_similarities = image_features @ class_text_features.T

    logit_scale = model.logit_scale.exp()
    scaled_sims = logit_scale * pure_similarities
    probs = scaled_sims.softmax(dim=-1).cpu().numpy()
    y_true = labels.cpu().numpy()

    auc = roc_auc_score(y_true, probs, multi_class="ovr")
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_true)

    return {
        "auc": auc,
        "accuracy": acc,
    }

def calculate_similarity_gap(model, dataloader, device, similarity_calculator):
    """Compute the similarity gap using cosine similarity only."""
    model.eval()
    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_embeds = model.get_image_features(pixel_values=pixel_values)
            text_embeds = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

            all_image_features.append(image_embeds)
            all_text_features.append(text_embeds)

    if not all_image_features:
        return 0.0, 0.0, 0.0

    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    metrics = similarity_calculator.batch_calculate_metrics(
        text_features, image_features, metric='cosine'
    )

    return metrics['delta'], metrics['mean_cosine_pos'], metrics['mean_cosine_neg']

def train_itc_layered(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)

    model = apply_layer_freezing(
        model,
        text_unfreeze_layers=args.text_unfreeze_layers,
        vision_unfreeze_layers=args.vision_unfreeze_layers
    )
    model.to(device)

    similarity_calculator = SimilarityMetricsCalculator()

    augment = create_augmentation() if args.use_augmentation else None

    collator = CLIPContrastiveDataCollator(processor)

    cfg = DATASET_CONFIGS[args.dataset]
    class_prompts = cfg["class_prompts"]
    data_root = args.data_root if args.data_root else cfg["data_root"]

    shot_values = [int(s) for s in args.shots.split(",")]
    run_id = args.run_id

    text_layers_str = str(args.text_unfreeze_layers)
    vision_layers_str = str(args.vision_unfreeze_layers)

    run_parent_dir = os.path.join(args.output_dir, f"run{run_id}_text{text_layers_str}_vis{vision_layers_str}")
    os.makedirs(run_parent_dir, exist_ok=True)

    for shot in shot_values:
        print(f"Run {run_id} | shot {shot}")

        train_csv = f"{data_root}/run_{run_id:02d}/train_shot_{shot}.csv"
        val_csv = f"{data_root}/val.csv"

        if not os.path.exists(train_csv):
            print(f"Warning: Training file not found: {train_csv}")
            continue

        train_dataset = CSVImageDataset(train_csv, processor, transform=augment, class_prompts=class_prompts)
        train_eval_dataset = CSVImageDataset(train_csv, processor, transform=None, class_prompts=class_prompts)
        val_dataset = CSVImageDataset(val_csv, processor, transform=None, class_prompts=class_prompts)

        shot_output_dir = os.path.join(run_parent_dir, f"shot{shot}_layers{text_layers_str}_{vision_layers_str}")
        os.makedirs(shot_output_dir, exist_ok=True)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )

        train_eval_dataloader = DataLoader(
            train_eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
        )

        model = CLIPModel.from_pretrained(args.model_name)
        model = apply_layer_freezing(
            model,
            text_unfreeze_layers=args.text_unfreeze_layers,
            vision_unfreeze_layers=args.vision_unfreeze_layers
        )
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

        best_val_auc = 0.0

        class_names = train_dataset.class_names

        csv_path = os.path.join(shot_output_dir, "similarity_gap_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_delta", "train_mean_cosine_pos", "train_mean_cosine_neg",
                           "val_delta", "val_mean_cosine_pos", "val_mean_cosine_neg", "val_auc", "train_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Shot={shot} Epoch={epoch} [Train]"):
                optimizer.zero_grad()

                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_loss=False,
                )

                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                logit_scale = model.logit_scale.exp()

                loss, _, _ = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch["pixel_values"].size(0)

            avg_loss = running_loss / len(train_dataloader.dataset)

            scheduler.step()

            train_delta, train_pos, train_neg = calculate_similarity_gap(model, train_eval_dataloader, device, similarity_calculator)

            model.eval()

            with torch.no_grad():
                train_metrics = evaluate_with_class_prototypes(model, train_eval_dataloader, device, class_names, class_prompts)
                train_auc = train_metrics["auc"]
                train_acc = train_metrics["accuracy"]

                val_metrics = evaluate_with_class_prototypes(model, val_dataloader, device, class_names, class_prompts)
                val_auc = val_metrics["auc"]
                val_acc = val_metrics["accuracy"]
                val_delta, val_pos, val_neg = calculate_similarity_gap(model, val_dataloader, device, similarity_calculator)

                current_logit_scale = model.logit_scale.exp().item()

                print(f"[Run={run_id} Shot={shot} Epoch={epoch}] loss={avg_loss:.4f} logit_scale={current_logit_scale:.2f} | "
                      f"Train: AUC={train_auc:.4f}, ACC={train_acc:.4f} | "
                      f"Val: AUC={val_auc:.4f}, ACC={val_acc:.4f} | "
                      f"Gap: train={train_delta:.4f} val={val_delta:.4f}")

                is_best = val_auc > best_val_auc
                if is_best:
                    best_val_auc = val_auc

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_delta, train_pos, train_neg,
                               val_delta, val_pos, val_neg, val_auc, avg_loss])

            if args.save_each_epoch:
                epoch_save_dir = os.path.join(shot_output_dir, f"epoch_{epoch}")
                os.makedirs(epoch_save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(epoch_save_dir, "model.pth"))

            if is_best:
                best_save_dir = os.path.join(shot_output_dir, "best_model")
                os.makedirs(best_save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_save_dir, "model.pth"))
                print(f"  -> Saved best model for shot={shot} to {best_save_dir} (val_auc={val_auc:.4f})")

            if early_stopping and early_stopping(val_auc, model):
                print(f"  -> Early stopping triggered at epoch {epoch} (val_auc={val_auc:.4f})")
                early_stopping.restore_best(model)
                break

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP ITC training with layered unfreezing and Similarity Gap analysis")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID (1-10)")
    parser.add_argument("--shots", type=str, default="1", help="Shots per class, comma-separated, e.g. '1,2,4'")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--output_dir", type=str, default="output_itc", help="Output directory")
    parser.add_argument("--model_name", type=str, default="vinid/plip", help="CLIP model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--save_each_epoch", action="store_true", help="Save model for each epoch")
    parser.add_argument("--text_unfreeze_layers", type=int, default=2,
                       help="Number of last layers to unfreeze in text encoder")
    parser.add_argument("--vision_unfreeze_layers", type=int, default=4,
                       help="Number of last layers to unfreeze in vision encoder")
    parser.add_argument("--dataset", type=str, default="vandy", choices=["vandy", "cornell"],
                        help="Dataset name used to select built-in data_root and class prompts")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root directory for dataset (overrides --dataset default)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_itc_layered(args)

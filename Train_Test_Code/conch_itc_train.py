#!/usr/bin/env python3
"""
CONCH few-shot ITC training.

This script uses CSV-based data splits, prototype evaluation, and
similarity-gap tracking across multiple shot settings.
"""

import os
import sys
import argparse
import random
import json
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

# CONCH imports
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

# Add local paths for augmentation and similarity utilities.
_THIS_DIR = Path(__file__).resolve()
_FEWSHOT_ROOT = next((p for p in _THIS_DIR.parents if p.name == "fewshot"), _THIS_DIR.parent)
_MODEL_ROOT = _FEWSHOT_ROOT / "model"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.append(str(_MODEL_ROOT))

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

def set_seed(seed: int):
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_roc_auc(y_true, probs):
    """Compute ROC AUC and return NaN if evaluation fails."""
    try:
        return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except:
        return float("nan")

class CSVImageDataset(Dataset):
    """CSV-based dataset loader."""
    
    def __init__(self, csv_path: str, preprocess, transform=None, class_prompts=None):
        self.csv_path = csv_path
        self.preprocess = preprocess
        self.transform = transform
        self.class_prompts = class_prompts or {}
        
        self.data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row['class_name']
                self.data.append({
                    'path': row['path'],
                    'class_name': class_name
                })
        
        # Build the class index mapping from the CSV file.
        unique_classes = sorted(set(item['class_name'] for item in self.data))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.class_names = unique_classes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['path']
        class_name = item['class_name']
        label = self.class_to_idx[class_name]
        
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_array = np.array(image)
            image_array = self.transform(image=image_array)["image"]
            image = Image.fromarray(image_array)

        prompt = self.class_prompts[class_name]
        
        return {
            "image": image,
            "label": label,
            "prompt": prompt,
            "path": image_path,
            "class_name": class_name
        }

class ITCCollator:
    """Collator for ITC training."""
    
    def __init__(self, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __call__(self, features):
        images = [self.preprocess(f["image"]) for f in features]

        prompts = [f["prompt"] for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        paths = [f["path"] for f in features]
        class_names = [f["class_name"] for f in features]

        text_tokens = tokenize(texts=prompts, tokenizer=self.tokenizer)
        
        return {
            "images": torch.stack(images),
            "text_tokens": text_tokens,
            "labels": labels,
            "paths": paths,
            "class_names": class_names
        }

class BalancedBatchSampler:
    """Ensure that each batch contains one sample from each class."""
    
    def __init__(self, dataset, batch_size=5, seed=42):
        if batch_size != 5:
            raise ValueError("BalancedBatchSampler requires batch_size=5 for 5-class balanced sampling")
        
        self.batch_size = batch_size
        self.dataset = dataset
        self.rng = random.Random(seed)
        
        self.class_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            label = dataset.class_to_idx[item['class_name']]
            self.class_to_indices[label].append(idx)

        if len(self.class_to_indices) != 5:
            raise ValueError(f"Expected 5 classes, got {len(self.class_to_indices)}")

        for label, indices in self.class_to_indices.items():
            if len(indices) == 0:
                class_name = dataset.idx_to_class[label]
                raise ValueError(f"Class {class_name} has no samples")

    def __iter__(self):
        min_samples_per_class = min(len(indices) for indices in self.class_to_indices.values())
        num_batches = min_samples_per_class

        shuffled_indices = {}
        for label, indices in self.class_to_indices.items():
            shuffled = indices.copy()
            self.rng.shuffle(shuffled)
            shuffled_indices[label] = shuffled

        for batch_idx in range(num_batches):
            batch = []
            for label in sorted(self.class_to_indices.keys()):
                sample_idx = shuffled_indices[label][batch_idx % len(shuffled_indices[label])]
                batch.append(sample_idx)
            yield batch

    def __len__(self):
        min_samples_per_class = min(len(indices) for indices in self.class_to_indices.values())
        return min_samples_per_class

class EarlyStopping:
    """Early stopping based on validation AUC."""
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, metric, model):
        stop = False
        if metric is None or metric != metric:
            self.counter += 1
        elif metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            stop = True
            
        return stop

    def restore(self, model):
        """Restore the best model state."""
        if self.best_state:
            model.load_state_dict(self.best_state)

def get_class_text_features(model, tokenizer, class_names, device, class_prompts):
    """Encode all class prompts into text features."""
    all_prompts = [class_prompts[cls] for cls in class_names]
    text_tokens = tokenize(texts=all_prompts, tokenizer=tokenizer)

    if isinstance(text_tokens, dict):
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    else:
        text_tokens = text_tokens.to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    return text_features

def calculate_similarity_gap(model, dataloader, device, similarity_calculator):
    """Compute the similarity gap with cosine similarity."""
    model.eval()
    total = 0
    sum_delta = 0.0
    sum_pos = 0.0
    sum_neg = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            text_tokens = batch["text_tokens"]
            if isinstance(text_tokens, dict):
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            else:
                text_tokens = text_tokens.to(device)

            img_feats = model.encode_image(images)
            txt_feats = model.encode_text(text_tokens)

            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)

            metrics = similarity_calculator.batch_calculate_metrics(
                txt_feats, img_feats, metric="cosine"
            )
            bs = metrics["positive_stats"]["count"]
            if bs == 0:
                continue

            total += bs
            sum_delta += metrics["delta"] * bs
            sum_pos += metrics["mean_cosine_pos"] * bs
            sum_neg += metrics["mean_cosine_neg"] * bs

    if total == 0:
        return 0.0, 0.0, 0.0

    return sum_delta / total, sum_pos / total, sum_neg / total

def evaluate_model_metrics(model, tokenizer, val_loader, class_names, device, class_prompts):
    """Evaluate the model and return AUC and accuracy."""
    model.eval()
    class_text_features = get_class_text_features(model, tokenizer, class_names, device, class_prompts)

    y_true, y_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            image_features = model.encode_image(images)
            pure_sims = image_features @ class_text_features.T
            logit_scale = getattr(model, 'logit_scale', torch.tensor(1.0, device=device)).exp()
            probs = (logit_scale * pure_sims).softmax(dim=-1)

            y_true.extend(labels.cpu().tolist())
            y_probs.extend(probs.cpu().tolist())

    auc = safe_roc_auc(y_true, y_probs)
    preds = np.argmax(np.array(y_probs), axis=-1) if y_probs else []
    acc = accuracy_score(y_true, preds) if len(preds) > 0 else 0.0
    return {"auc": auc, "acc": acc}

def train_conch_itc(args):
    """Run CONCH ITC training across the requested shot settings."""
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16',
        'hf_hub:MahmoodLab/conch',
        device=device
    )
    tokenizer = get_tokenizer()
    similarity_calculator = SimilarityMetricsCalculator()

    transform = create_augmentation() if args.use_augmentation else None

    cfg = DATASET_CONFIGS[args.dataset]
    class_prompts = cfg["class_prompts"]
    data_root = args.data_root if args.data_root else cfg["data_root"]

    shots_list = [int(s.strip()) for s in args.shots.split(',')]

    for shot_count in shots_list:
        print(f"Run {args.run_id} | shot {shot_count}")

        text_layers_str = "-".join(str(l) for l in args.text_unfreeze_layers)
        vision_blocks_str = "-".join(str(b) for b in args.vision_unfreeze_blocks)

        run_name = f"run{args.run_id}_itc_text_{text_layers_str}_vision_{vision_blocks_str}"
        if args.use_augmentation:
            run_name += "_aug"

        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        exp_dir = os.path.join(run_dir, f"shot{shot_count}_itc")
        os.makedirs(exp_dir, exist_ok=True)

        train_csv = os.path.join(data_root, f"run_{args.run_id:02d}", f"train_shot_{shot_count}.csv")
        val_csv = os.path.join(data_root, "val.csv")

        train_dataset = CSVImageDataset(train_csv, preprocess, transform, class_prompts=class_prompts)
        val_dataset = CSVImageDataset(val_csv, preprocess, class_prompts=class_prompts)

        collator = ITCCollator(preprocess, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=BalancedBatchSampler(train_dataset, batch_size=5, seed=args.seed),
            collate_fn=collator,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True
        )

        class_names = train_dataset.class_names

        missing_prompts = set(class_names) - set(class_prompts.keys())
        if missing_prompts:
            raise ValueError(f"Missing class prompts: {missing_prompts}")

        model, _ = create_model_from_pretrained(
            'conch_ViT-B-16',
            'hf_hub:MahmoodLab/conch',
            device=device
        )

        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'text') and hasattr(model.text, 'transformer'):
            for layer_idx in args.text_unfreeze_layers:
                if layer_idx < len(model.text.transformer.resblocks):
                    for param in model.text.transformer.resblocks[layer_idx].parameters():
                        param.requires_grad = True
                else:
                    print(f"Warning: text layer {layer_idx} does not exist")
        else:
            print("Warning: text model structure not found")

        if hasattr(model, 'visual') and hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks'):
            for block_idx in args.vision_unfreeze_blocks:
                if block_idx < len(model.visual.trunk.blocks):
                    for param in model.visual.trunk.blocks[block_idx].parameters():
                        param.requires_grad = True
                else:
                    print(f"Warning: vision block {block_idx} does not exist")
        else:
            print("Warning: vision model structure not found")

        if hasattr(model, 'logit_scale'):
            model.logit_scale.requires_grad = True
        else:
            print("Warning: logit_scale not found")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

        similarity_metrics_path = os.path.join(exp_dir, "metrics.csv")
        with open(similarity_metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss",
                "train_delta", "train_mean_cosine_pos", "train_mean_cosine_neg",
                "val_delta", "val_mean_cosine_pos", "val_mean_cosine_neg",
                "val_auc", "val_acc"
            ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in progress_bar:
                optimizer.zero_grad()

                images = batch["images"].to(device)

                if isinstance(batch["text_tokens"], dict):
                    text_tokens = {k: v.to(device) for k, v in batch["text_tokens"].items()}
                else:
                    text_tokens = batch["text_tokens"].to(device)

                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)

                logit_scale = getattr(model, 'logit_scale', torch.tensor(1.0, device=device)).exp()
                similarities = logit_scale * (image_features @ text_features.T)

                batch_size = images.size(0)
                targets = torch.arange(batch_size, device=device)

                loss_i2t = F.cross_entropy(similarities, targets)
                loss_t2i = F.cross_entropy(similarities.T, targets)
                loss = (loss_i2t + loss_t2i) / 2

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_loss / max(num_batches, 1)
            train_gap, train_mean_pos, train_mean_neg = calculate_similarity_gap(
                model, train_loader, device, similarity_calculator
            )
            val_gap, val_mean_pos, val_mean_neg = calculate_similarity_gap(
                model, val_loader, device, similarity_calculator
            )

            val_metrics = evaluate_model_metrics(model, tokenizer, val_loader, class_names, device, class_prompts)
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]

            print(f"[Run={args.run_id} Shot={shot_count} Epoch={epoch}] loss={avg_train_loss:.4f} | "
                  f"Val: AUC={val_auc:.4f} ACC={val_acc:.4f} | Gap: train={train_gap:.4f} val={val_gap:.4f}")

            is_best = val_auc > early_stopping.best

            with open(similarity_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, avg_train_loss,
                    train_gap, train_mean_pos, train_mean_neg,
                    val_gap, val_mean_pos, val_mean_neg,
                    val_auc, val_acc
                ])

            if is_best:
                best_model_dir = os.path.join(exp_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pth"))
                print(f"  -> Saved best model to {best_model_dir}")

            if early_stopping.step(val_auc, model):
                print(f"  -> Early stopping triggered at epoch {epoch} (val_auc={val_auc:.4f})")
                early_stopping.restore(model)
                break

        config = {
            "run_id": args.run_id,
            "shot_count": shot_count,
            "text_unfreeze_layers": args.text_unfreeze_layers,
            "vision_unfreeze_blocks": args.vision_unfreeze_blocks,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "use_augmentation": args.use_augmentation,
            "seed": args.seed,
            "data_root": data_root,
            "final_val_auc": val_auc if 'val_auc' in locals() else 0.0,
            "best_val_auc": early_stopping.best
        }

        with open(os.path.join(exp_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CONCH few-shot ITC training")

    parser.add_argument("--run_id", type=int, default=1,
                        help="Split identifier (1-10)")
    parser.add_argument("--dataset", type=str, default="vandy", choices=["vandy", "cornell"],
                        help="Dataset name used to select built-in data_root and class prompts")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root containing val.csv/test.csv and run_{id}/train_shot_{shot}.csv (overrides --dataset default)")
    parser.add_argument("--output_dir", type=str, default="./itc_experiments",
                        help="Output directory")

    parser.add_argument("--shots", type=str, default="1,2,4,8,16,32",
                        help="Few-shot values per class, comma-separated, e.g. '1,2,4'")
    parser.add_argument("--text_unfreeze_layers", type=int, nargs='+', default=[10, 11],
                        help="Text layer indices to unfreeze")
    parser.add_argument("--vision_unfreeze_blocks", type=int, nargs='+', default=[10, 11],
                        help="Vision block indices to unfreeze")

    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Validation batch size")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")

    parser.add_argument("--use_augmentation", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_conch_itc(args)

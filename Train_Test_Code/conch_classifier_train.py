import os
import sys
import random
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained
from sklearn.metrics import roc_auc_score, accuracy_score

# Add the local model directory for augmentation utilities.
_THIS_DIR = Path(__file__).resolve()
_FEWSHOT_ROOT = next((p for p in _THIS_DIR.parents if p.name == "fewshot"), _THIS_DIR.parent)
_MODEL_ROOT = _FEWSHOT_ROOT / "model"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.append(str(_MODEL_ROOT))
from augmentations import create_augmentation

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
    def __init__(self, csv_path, preprocess=None, transform=None, class_prompts=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.transform = transform
        self.class_prompts = class_prompts or {}

        unique_classes = sorted(self.df['class_name'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.class_names = unique_classes

        missing = set(unique_classes) - set(self.class_prompts.keys())
        if missing:
            raise ValueError(f"These classes are missing in class_prompts: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        class_name = row['class_name']
        label = self.class_to_idx[class_name]
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_np = np.array(image)
            image = self.transform(image=image_np)["image"]
            image = Image.fromarray(image.astype(np.uint8))

        if self.preprocess:
            image = self.preprocess(image)

        return {
            "image": image,
            "labels": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
            "class_name": class_name
        }

class MLPBatchNormClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class FrozenCONCHWithClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, dropout=0.5, device='cuda:0'):
        super().__init__()
        self.device = device

        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            'hf_hub:MahmoodLab/conch',
            device=device
        )
        
        self.backbone = model
        self.preprocess = preprocess

        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_dim = 512
        self.classifier = MLPBatchNormClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, images):
        with torch.no_grad():
            image_features = self.backbone.encode_image(images)

        logits = self.classifier(image_features)
        return logits, image_features.detach()

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

def evaluate_classifier(model, dataloader, device, class_names):
    """Evaluate classifier performance on a dataloader."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model(images)
            
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = F.softmax(logits, dim=-1).cpu().numpy()
    y_true = labels.cpu().numpy()

    auc = roc_auc_score(y_true, probs, multi_class="ovr")
    preds = probs.argmax(axis=-1)
    acc = accuracy_score(y_true, preds)
    
    return {
        "auc": auc,
        "accuracy": acc,
    }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    image_paths = [item["image_path"] for item in batch]
    class_names = [item["class_name"] for item in batch]
    
    return {
        "image": images,
        "labels": labels,
        "image_path": image_paths,
        "class_name": class_names
    }

def train_few_shot(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    augment = create_augmentation() if args.use_augmentation else None

    cfg = DATASET_CONFIGS[args.dataset]
    class_prompts = cfg["class_prompts"]
    data_root = args.data_root if args.data_root else cfg["data_root"]

    shot_values = [int(s) for s in args.shots.split(",")]
    run_id = args.run_id

    run_parent_dir = os.path.join(
        args.output_dir,
        f"run{run_id}_hd{args.hidden_dim}_drop{args.dropout}_lr{args.lr}"
    )
    os.makedirs(run_parent_dir, exist_ok=True)

    for shot in shot_values:
        print(f"Run {run_id} | shot {shot}")

        train_csv = os.path.join(data_root, f"run_{run_id:02d}", f"train_shot_{shot}.csv")
        val_csv = os.path.join(data_root, "val.csv")

        if not os.path.exists(train_csv):
            print(f"Warning: training file not found: {train_csv}")
            continue
        if not os.path.exists(val_csv):
            print(f"Warning: validation file not found: {val_csv}")
            continue

        temp_model, temp_preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            'hf_hub:MahmoodLab/conch',
            device=device
        )

        train_dataset = CSVImageDataset(train_csv, preprocess=temp_preprocess, transform=augment, class_prompts=class_prompts)
        val_dataset = CSVImageDataset(val_csv, preprocess=temp_preprocess, transform=None, class_prompts=class_prompts)

        del temp_model
        torch.cuda.empty_cache()

        shot_output_dir = os.path.join(run_parent_dir, f"shot{shot}_mlp")
        os.makedirs(shot_output_dir, exist_ok=True)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

        num_classes = len(train_dataset.class_names)
        model = FrozenCONCHWithClassifier(
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            device=device
        )
        model.to(device)

        # Optimizer: update only the classifier head.
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

        best_val_auc = float("-inf")
        best_epoch = 0

        class_names = train_dataset.class_names

        csv_path = os.path.join(shot_output_dir, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_auc", "val_accuracy"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Shot={shot} Epoch={epoch} [Train]"):
                optimizer.zero_grad()

                images = batch["image"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(images)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            avg_loss = running_loss / len(train_dataloader.dataset)

            model.eval()

            with torch.no_grad():
                val_metrics = evaluate_classifier(model, val_dataloader, device, class_names)
                val_auc = val_metrics["auc"]
                val_acc = val_metrics["accuracy"]

                print(f"[Run={run_id} Shot={shot} Epoch={epoch}] loss={avg_loss:.4f} | "
                      f"Val: AUC={val_auc:.4f}, ACC={val_acc:.4f}")

                monitor_metric = val_auc
                is_best = val_auc > best_val_auc
                if is_best:
                    best_val_auc = val_auc
                    best_epoch = epoch

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss, val_auc, val_acc])

            if is_best:
                best_save_dir = os.path.join(shot_output_dir, "best_model")
                os.makedirs(best_save_dir, exist_ok=True)
                torch.save(model.classifier.state_dict(), os.path.join(best_save_dir, "classifier.pth"))
                print(f"  -> Saved best classifier for shot={shot} to {best_save_dir} (val_auc={monitor_metric:.4f})")

            if early_stopping and early_stopping(monitor_metric, model):
                print(f"  -> Early stopping triggered at epoch {epoch} "
                      f"(best_epoch={best_epoch}, best_val_auc={best_val_auc:.4f})")
                early_stopping.restore_best(model)
                break

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot CONCH MLPBatchNormClassifier training with frozen backbone")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID (1-10) for selecting training data")
    parser.add_argument("--shots", type=str, default="1", help="Comma-separated few-shot values per class, e.g., '1,2,4'")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="output_conch_classifiers")
    parser.add_argument("--dataset", type=str, default="vandy", choices=["vandy", "cornell"],
                        help="Dataset name used to select built-in data_root and class prompts")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root containing val.csv/test.csv and run_{id}/train_shot_{shot}.csv (overrides --dataset default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for MLP classifier")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_few_shot(args)

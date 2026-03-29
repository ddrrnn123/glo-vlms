#!/usr/bin/env python3

import os
import sys
import argparse
import json
import random
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from PIL import Image

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            raise ValueError(f"Missing classes in class_prompts: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        class_name = row['class_name']
        label = self.class_to_idx[class_name]
        
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image_np = np.array(image)
            image = self.transform(image=image_np)["image"]
            image = Image.fromarray(image.astype(np.uint8))

        if self.preprocess:
            image = self.preprocess(image)

        text_prompt = self.class_prompts[class_name]

        return {
            "image": image,
            "text_prompt": text_prompt,
            "labels": torch.tensor(label, dtype=torch.long),
            "image_path": image_path,
            "class_name": class_name
        }

class CONCHContrastiveDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        
        images = torch.stack([f["image"] for f in features])
        batch["images"] = images

        text_prompts = [f["text_prompt"] for f in features]
        text_tokens = tokenize(texts=text_prompts, tokenizer=self.tokenizer)
        batch["text_tokens"] = text_tokens

        batch["labels"] = torch.stack([f["labels"] for f in features])
        batch["image_paths"] = [f["image_path"] for f in features]
        batch["class_names"] = [f["class_name"] for f in features]

        return batch

class LoRACONCHModel(torch.nn.Module):
    def __init__(self, lora_config, target_modalities=["vision"], device='cuda:1'):
        super().__init__()
        self.device = device
        self.target_modalities = target_modalities

        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            'hf_hub:MahmoodLab/conch',
            device=device
        )
        
        self.conch_model = model
        self.preprocess = preprocess
        self.tokenizer = get_tokenizer()

        for param in self.conch_model.parameters():
            param.requires_grad = False

        self.conch_model = get_peft_model(self.conch_model, lora_config)

        if hasattr(self.conch_model, 'logit_scale'):
            self.conch_model.logit_scale.requires_grad_(True)

    def forward(self, **inputs):
        """Forward pass for contrastive training."""
        images = inputs.get('images')
        text_tokens = inputs.get('text_tokens')

        if isinstance(text_tokens, dict):
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        else:
            text_tokens = text_tokens.to(self.device)

        image_embeds = self.conch_model.encode_image(images)
        text_embeds = self.conch_model.encode_text(text_tokens)

        logit_scale = getattr(self.conch_model, 'logit_scale', torch.tensor(1.0, device=self.device)).exp()
        logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
        logits_per_text = logit_scale * (text_embeds @ image_embeds.T)

        batch_size = image_embeds.size(0)
        contrastive_labels = torch.arange(batch_size, device=self.device)
        loss_i2t = F.cross_entropy(logits_per_image, contrastive_labels)
        loss_t2i = F.cross_entropy(logits_per_text, contrastive_labels)
        loss = (loss_i2t + loss_t2i) / 2

        return {"loss": loss}

    def get_image_features(self, images):
        """Return normalized image features."""
        return self.conch_model.encode_image(images)

    def get_text_features(self, text_tokens):
        """Return normalized text features."""
        if isinstance(text_tokens, dict):
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        else:
            text_tokens = text_tokens.to(self.device)
        return self.conch_model.encode_text(text_tokens)

def get_conch_lora_targets(target_modalities, vision_layers, text_layers):
    """Build LoRA target module names for the requested CONCH layers."""
    target_modules = []

    if "vision" in target_modalities:
        for layer in vision_layers:
            target_modules.extend([
                f"visual.trunk.blocks.{layer}.attn.qkv",
                f"visual.trunk.blocks.{layer}.attn.proj",
            ])

    if "text" in target_modalities:
        for layer in text_layers:
            target_modules.extend([
                f"text.transformer.resblocks.{layer}.attn.out_proj",
                f"text.transformer.resblocks.{layer}.mlp.c_fc",
            ])
    
    return target_modules

def get_class_text_features(model, tokenizer, class_names, device, class_prompts):
    """Encode class prompts into text features."""
    with torch.no_grad():
        prompts = [class_prompts[c] for c in class_names]
        text_tokens = tokenize(texts=prompts, tokenizer=tokenizer)

        if isinstance(text_tokens, dict):
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        else:
            text_tokens = text_tokens.to(device)

        class_text_features = model.get_text_features(text_tokens)
    return class_text_features

def evaluate_with_class_prototypes(model, dataloader, device, class_names, class_prompts):
    """Evaluate using class-prompt prototypes."""
    model.eval()
    all_image_features = []
    all_labels = []
    all_pure_sims = []

    tokenizer = model.tokenizer
    class_text_features = get_class_text_features(model, tokenizer, class_names, device, class_prompts)

    for batch in dataloader:
        with torch.no_grad():
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            image_features = model.get_image_features(images)

            pure_sims = image_features @ class_text_features.T
            all_image_features.append(image_features)
            all_labels.append(labels)
            all_pure_sims.append(pure_sims)

    image_features = torch.cat(all_image_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    pure_similarities = torch.cat(all_pure_sims, dim=0)

    logit_scale = getattr(model.conch_model, 'logit_scale', torch.tensor(1.0, device=device)).exp()
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
            images = batch["images"].to(device)
            text_tokens = batch["text_tokens"]

            if isinstance(text_tokens, dict):
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            else:
                text_tokens = text_tokens.to(device)

            image_embeds = model.get_image_features(images)
            text_embeds = model.get_text_features(text_tokens)

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
def save_model_with_logit_scale(model, save_dir):
    """Save the LoRA adapter together with logit_scale."""
    os.makedirs(save_dir, exist_ok=True)

    model.conch_model.save_pretrained(save_dir)

    if hasattr(model.conch_model, 'logit_scale'):
        torch.save(
            {"logit_scale": model.conch_model.logit_scale.detach().cpu()},
            os.path.join(save_dir, "extra_logit_scale.pt")
        )
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
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

def train_few_shot(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    similarity_calculator = SimilarityMetricsCalculator()

    augment = create_augmentation() if args.use_augmentation else None

    cfg = DATASET_CONFIGS[args.dataset]
    class_prompts = cfg["class_prompts"]
    data_root = args.data_root if args.data_root else cfg["data_root"]

    shot_values = [int(s) for s in args.shots.split(",")]
    run_id = args.run_id
    target_modalities = args.target_modalities.split(",")

    modalities_str = "_".join(target_modalities)
    lora_config_str = f"r{args.lora_r}_alpha{args.lora_alpha}_{modalities_str}"

    run_parent_dir = os.path.join(args.output_dir, f"run{run_id}_lora_{lora_config_str}")
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
        train_eval_dataset = CSVImageDataset(train_csv, preprocess=temp_preprocess, transform=None, class_prompts=class_prompts)
        val_dataset = CSVImageDataset(val_csv, preprocess=temp_preprocess, transform=None, class_prompts=class_prompts)

        del temp_model
        torch.cuda.empty_cache()

        shot_output_dir = os.path.join(run_parent_dir, f"shot{shot}_lora")
        os.makedirs(shot_output_dir, exist_ok=True)

        tokenizer = get_tokenizer()
        collator = CONCHContrastiveDataCollator(tokenizer)

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

        vision_layers = [int(x) for x in args.vision_layers.split(",")]
        text_layers = [int(x) for x in args.text_layers.split(",")]

        target_modules = get_conch_lora_targets(target_modalities, vision_layers, text_layers)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )

        model = LoRACONCHModel(
            lora_config=lora_config,
            target_modalities=target_modalities,
            device=device
        )
        model.to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, 
            weight_decay=args.weight_decay
        )

        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

        best_val_auc = float("-inf")

        class_names = train_dataset.class_names

        csv_path = os.path.join(shot_output_dir, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss",
                "train_delta", "train_mean_cosine_pos", "train_mean_cosine_neg",
                "val_delta", "val_mean_cosine_pos", "val_mean_cosine_neg",
                "val_auc", "val_acc"
            ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Shot={shot} Epoch={epoch} [Train]"):
                optimizer.zero_grad()

                images = batch["images"].to(device)
                text_tokens = batch["text_tokens"]

                outputs = model(images=images, text_tokens=text_tokens)
                loss = outputs["loss"]

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            avg_loss = running_loss / len(train_dataloader.dataset)

            model.eval()

            with torch.no_grad():
                train_delta, train_pos, train_neg = calculate_similarity_gap(
                    model, train_eval_dataloader, device, similarity_calculator
                )

                val_metrics = evaluate_with_class_prototypes(model, val_dataloader, device, class_names, class_prompts)
                val_auc = val_metrics["auc"]
                val_acc = val_metrics["accuracy"]

                val_delta, val_pos, val_neg = calculate_similarity_gap(
                    model, val_dataloader, device, similarity_calculator
                )

                current_logit_scale = getattr(model.conch_model, 'logit_scale', torch.tensor(1.0, device=device)).exp().item()

                print(f"[Run={run_id} Shot={shot} Epoch={epoch}] loss={avg_loss:.4f} logit_scale={current_logit_scale:.2f} | "
                      f"Val: AUC={val_auc:.4f}, ACC={val_acc:.4f} | "
                      f"Gap: train={train_delta:.4f} val={val_delta:.4f}")

                is_best = val_auc > best_val_auc
                if is_best:
                    best_val_auc = val_auc

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss,
                               train_delta, train_pos, train_neg, 
                               val_delta, val_pos, val_neg, val_auc, val_acc])

            if is_best:
                best_save_dir = os.path.join(shot_output_dir, "best_model")
                save_model_with_logit_scale(model, best_save_dir)
                print(f"  -> Saved best LoRA model and logit_scale for shot={shot} to {best_save_dir} (val_auc={val_auc:.4f})")

            if early_stopping and early_stopping(val_auc, model):
                print(f"  -> Early stopping triggered at epoch {epoch} (val_auc={val_auc:.4f})")
                early_stopping.restore_best(model)
                break

        config = {
            "run_id": run_id,
            "shot": shot,
            "data_root": data_root,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modalities": target_modalities,
            "vision_layers": args.vision_layers,
            "text_layers": args.text_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "use_augmentation": args.use_augmentation,
            "seed": args.seed,
            "best_val_auc": best_val_auc,
        }
        with open(os.path.join(shot_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="CONCH Few-shot LoRA Fine-tuning with Vision and Text Support")

    parser.add_argument("--run_id", type=int, required=True, help="Run ID (1-10) for selecting training data")
    parser.add_argument("--shots", type=str, default="1", help="Comma-separated few-shot values per class, e.g., '1,2,4'")
    parser.add_argument("--dataset", type=str, default="vandy", choices=["vandy", "cornell"],
                        help="Dataset name used to select built-in data_root and class prompts")
    parser.add_argument("--data_root", type=str, default="",
                        help="Root containing val.csv/test.csv and run_{id}/train_shot_{shot}.csv (overrides --dataset default)")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="output_conch_lora_unified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--save_each_epoch", action="store_true", help="Save LoRA adapter for each epoch")

    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modalities", type=str, default="vision", help="Target modalities: 'vision', 'text', or 'vision,text'")
    parser.add_argument("--vision_layers", type=str, default="10,11", help="Vision transformer layers for LoRA, e.g., '10,11'")
    parser.add_argument("--text_layers", type=str, default="10,11", help="Text transformer layers for LoRA, e.g., '10,11'")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_few_shot(args)

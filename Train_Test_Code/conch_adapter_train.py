import os
import sys
import argparse
import random
import csv
import math
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, "..", "..", "model")))
from augmentations import create_augmentation

sys.path.append("/Data3/Daniel")
from similarity_metrics import SimilarityMetricsCalculator


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


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, preprocess, transform=None, class_prompts=None):
        self.df = pd.read_csv(csv_path)
        self.preprocess = preprocess
        self.transform = transform
        self.class_prompts = class_prompts or {}

        unique_classes = sorted(self.df["class_name"].unique())
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
        image_path = row["path"]
        class_name = row["class_name"]
        label = self.class_to_idx[class_name]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        image = Image.fromarray(image)

        return {
            "image": self.preprocess(image),
            "label": torch.tensor(label, dtype=torch.long),
            "prompt": self.class_prompts[class_name],
            "image_path": image_path,
        }


class CONCHContrastiveDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        images = torch.stack([f["image"] for f in features])
        labels = torch.stack([f["label"] for f in features])
        prompts = [f["prompt"] for f in features]
        image_paths = [f["image_path"] for f in features]
        text_tokens = tokenize(texts=prompts, tokenizer=self.tokenizer)

        return {
            "images": images,
            "text_tokens": text_tokens,
            "labels": labels,
            "image_paths": image_paths,
        }


class CONCHWithAdapters(nn.Module):
    def __init__(self, base_model, adapter_dim=64, insert_vision_layers=None, insert_text_layers=None):
        super().__init__()
        self.base_model = base_model

        for p in self.base_model.parameters():
            p.requires_grad = False
        if hasattr(self.base_model, "logit_scale"):
            self.base_model.logit_scale.requires_grad = True

        self.vision_adapters = None
        self.vision_target_ids = set()
        self.vision_original_forwards = {}

        if insert_vision_layers:
            vision = self.base_model.visual
            vision_d_model = getattr(vision.trunk.norm, "normalized_shape", [768])[0]
            self.vision_num_blocks = len(vision.trunk.blocks)
            valid_vision_layers = [i for i in insert_vision_layers if 0 <= i < self.vision_num_blocks]
            self.vision_target_ids = set(valid_vision_layers)
            if self.vision_target_ids:
                self.vision_adapters = nn.ModuleDict(
                    {str(i): BottleneckAdapter(vision_d_model, adapter_dim) for i in self.vision_target_ids}
                )
                self._monkey_patch_vision_blocks(vision.trunk.blocks)

        self.text_adapters = None
        self.text_target_ids = set()
        self.text_original_forwards = {}

        if insert_text_layers and hasattr(self.base_model, "text") and hasattr(self.base_model.text, "transformer"):
            text_transformer = self.base_model.text.transformer
            if hasattr(text_transformer, "resblocks"):
                text_d_model = 768
                self.text_num_blocks = len(text_transformer.resblocks)
                valid_text_layers = [i for i in insert_text_layers if 0 <= i < self.text_num_blocks]
                self.text_target_ids = set(valid_text_layers)
                if self.text_target_ids:
                    self.text_adapters = nn.ModuleDict(
                        {str(i): BottleneckAdapter(text_d_model, adapter_dim) for i in self.text_target_ids}
                    )
                    self._monkey_patch_text_blocks(text_transformer.resblocks)

    def _monkey_patch_vision_blocks(self, vision_blocks):
        for layer_idx, block in enumerate(vision_blocks):
            if layer_idx in self.vision_target_ids:
                original_forward = block.forward
                self.vision_original_forwards[layer_idx] = original_forward

                def create_patched_forward(block_idx, orig_forward):
                    def patched_forward(x, *args, **kwargs):
                        output = orig_forward(x, *args, **kwargs)
                        if str(block_idx) in self.vision_adapters:
                            output = self.vision_adapters[str(block_idx)](output)
                        return output

                    return patched_forward

                block.forward = create_patched_forward(layer_idx, original_forward)

    def _monkey_patch_text_blocks(self, resblocks):
        for layer_idx, resblock in enumerate(resblocks):
            if layer_idx in self.text_target_ids:
                original_forward = resblock.forward
                self.text_original_forwards[layer_idx] = original_forward

                def create_patched_forward(block_idx, orig_forward):
                    def patched_forward(x, *args, **kwargs):
                        output = orig_forward(x, *args, **kwargs)
                        if str(block_idx) in self.text_adapters:
                            output = self.text_adapters[str(block_idx)](output)
                        return output

                    return patched_forward

                resblock.forward = create_patched_forward(layer_idx, original_forward)

    def _restore_vision_blocks(self):
        if hasattr(self.base_model, "visual") and hasattr(self.base_model.visual, "trunk"):
            vision_blocks = self.base_model.visual.trunk.blocks
            for layer_idx, original_forward in self.vision_original_forwards.items():
                if layer_idx < len(vision_blocks):
                    vision_blocks[layer_idx].forward = original_forward
        self.vision_original_forwards.clear()

    def _restore_text_blocks(self):
        if hasattr(self.base_model, "text") and hasattr(self.base_model.text, "transformer"):
            text_transformer = self.base_model.text.transformer
            if hasattr(text_transformer, "resblocks"):
                for layer_idx, original_forward in self.text_original_forwards.items():
                    if layer_idx < len(text_transformer.resblocks):
                        text_transformer.resblocks[layer_idx].forward = original_forward
        self.text_original_forwards.clear()

    def __del__(self):
        if hasattr(self, "vision_original_forwards"):
            self._restore_vision_blocks()
        if hasattr(self, "text_original_forwards"):
            self._restore_text_blocks()

    def encode_image(self, images_bchw):
        return F.normalize(self.base_model.encode_image(images_bchw), dim=-1)

    def encode_text(self, text_tokens: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return F.normalize(self.base_model.encode_text(text_tokens), dim=-1)

    def forward(self, images, text_tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        return image_features, text_features

    @property
    def logit_scale(self):
        return self.base_model.logit_scale

    def save_adapters(self, save_path):
        adapter_state = {}
        if self.vision_adapters:
            adapter_state["vision_adapters"] = self.vision_adapters.state_dict()
        if self.text_adapters:
            adapter_state["text_adapters"] = self.text_adapters.state_dict()
        torch.save(adapter_state, save_path)

    def load_adapters(self, load_path):
        adapter_state = torch.load(load_path, map_location="cpu")
        if "vision_adapters" in adapter_state and self.vision_adapters:
            self.vision_adapters.load_state_dict(adapter_state["vision_adapters"])
        if "text_adapters" in adapter_state and self.text_adapters:
            self.text_adapters.load_state_dict(adapter_state["text_adapters"])

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        if self.vision_adapters:
            self.vision_adapters.train(mode)
        if self.text_adapters:
            self.text_adapters.train(mode)
        return self


def move_tokens_to_device(tokens, device):
    if isinstance(tokens, dict):
        return {k: v.to(device) for k, v in tokens.items()}
    return tokens.to(device)


def get_class_text_features(model, tokenizer, class_names, device, class_prompts):
    with torch.no_grad():
        prompts = [class_prompts[c] for c in class_names]
        text_tokens = tokenize(texts=prompts, tokenizer=tokenizer)
        text_tokens = move_tokens_to_device(text_tokens, device)
        return model.encode_text(text_tokens)


def evaluate_all_metrics(model, dataloader, device, class_names, tokenizer, similarity_calculator, class_prompts):
    model.eval()
    all_image_features = []
    all_text_features = []
    all_labels = []

    class_text_features = get_class_text_features(model, tokenizer, class_names, device, class_prompts)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            text_tokens = move_tokens_to_device(batch["text_tokens"], device)
            labels = batch["labels"].to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            all_image_features.append(image_features)
            all_text_features.append(text_features)
            all_labels.append(labels)

    if not all_image_features:
        return {
            "auc": 0.0,
            "accuracy": 0.0,
            "delta": 0.0,
            "mean_cosine_pos": 0.0,
            "mean_cosine_neg": 0.0,
        }

    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    with torch.no_grad():
        pure_sims = image_features @ class_text_features.T
        scaled_sims = model.logit_scale.exp() * pure_sims
        probs = scaled_sims.softmax(dim=-1).cpu().numpy()
        y_true = labels.cpu().numpy()

    auc = roc_auc_score(y_true, probs, multi_class="ovr")
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_true)

    metrics = similarity_calculator.batch_calculate_metrics(text_features, image_features, metric="cosine")
    return {
        "auc": auc,
        "accuracy": acc,
        "delta": metrics["delta"],
        "mean_cosine_pos": metrics["mean_cosine_pos"],
        "mean_cosine_neg": metrics["mean_cosine_neg"],
    }


class UnifiedEvaluationCallback:
    def __init__(
        self,
        model,
        tokenizer,
        val_dataloader,
        class_names,
        output_dir,
        similarity_calculator,
        run_id,
        shot,
        adapter_config,
        class_prompts,
        patience=7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataloader = val_dataloader
        self.class_names = class_names
        self.output_dir = output_dir
        self.similarity_calculator = similarity_calculator
        self.run_id = run_id
        self.shot = shot
        self.adapter_config = adapter_config
        self.class_prompts = class_prompts
        self.best_val_auc = 0.0
        self.best_epoch = None
        self.patience = patience
        self.patience_counter = 0
        self.min_delta = 0.001

        self.csv_path = os.path.join(output_dir, "similarity_gap_metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_delta",
                    "train_mean_cosine_pos",
                    "train_mean_cosine_neg",
                    "val_delta",
                    "val_mean_cosine_pos",
                    "val_mean_cosine_neg",
                    "val_auc",
                    "train_loss",
                ]
            )

    def on_epoch_end(self, epoch, device, train_loss=0.0):
        val_metrics = evaluate_all_metrics(
            self.model,
            self.val_dataloader,
            device,
            self.class_names,
            self.tokenizer,
            self.similarity_calculator,
            self.class_prompts,
        )
        val_auc = val_metrics["auc"]
        val_acc = val_metrics["accuracy"]
        val_delta = val_metrics["delta"]
        val_pos = val_metrics["mean_cosine_pos"]
        val_neg = val_metrics["mean_cosine_neg"]
        current_logit_scale = self.model.logit_scale.exp().item()

        print(
            f"[Run={self.run_id} Shot={self.shot} Epoch={epoch}] "
            f"loss={train_loss:.4f} logit_scale={current_logit_scale:.2f} | "
            f"Val: AUC={val_auc:.4f}, ACC={val_acc:.4f}, Gap={val_delta:.4f}"
        )

        is_best = val_auc > self.best_val_auc + self.min_delta
        if is_best:
            self.best_val_auc = val_auc
            self.patience_counter = 0
            self.best_epoch = epoch
        else:
            self.patience_counter += 1

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 0, 0, 0, val_delta, val_pos, val_neg, val_auc, train_loss])

        if is_best:
            best_save_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_save_dir, exist_ok=True)
            best_adapter_weights_path = os.path.join(best_save_dir, "adapter_weights.pth")
            self.model.save_adapters(best_adapter_weights_path)
            print(
                f"  -> Saved best adapter (epoch {epoch}) for shot={self.shot} "
                f"to {best_save_dir} (val_auc={val_auc:.4f})"
            )

        should_stop = self.patience_counter >= self.patience
        if should_stop:
            print(f"  -> Early stopping triggered at epoch {epoch} (val_auc={val_auc:.4f})")

        return should_stop


def train_conch_adapter(args):
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer()
    _, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
    similarity_calculator = SimilarityMetricsCalculator()
    augment = create_augmentation() if args.use_augmentation else None
    collator = CONCHContrastiveDataCollator(tokenizer)

    insert_vision_layers = [int(x) for x in args.insert_vision_layers.split(",")] if args.insert_vision_layers else []
    insert_text_layers = [int(x) for x in args.insert_text_layers.split(",")] if args.insert_text_layers else []

    cfg = DATASET_CONFIGS[args.dataset]
    class_prompts = cfg["class_prompts"]
    data_root = args.data_root if args.data_root else cfg["data_root"]
    shot_values = [int(s) for s in args.shots.split(",")]
    run_id = args.run_id

    vision_layers_str = "_".join(map(str, insert_vision_layers)) if insert_vision_layers else "none"
    text_layers_str = "_".join(map(str, insert_text_layers)) if insert_text_layers else "none"
    run_parent_dir = os.path.join(
        args.output_dir,
        f"run{run_id}_vis{vision_layers_str}_text{text_layers_str}_dim{args.adapter_dim}",
    )
    os.makedirs(run_parent_dir, exist_ok=True)

    for shot in shot_values:
        print(f"Run {run_id} | shot {shot}")

        train_csv = f"{data_root}/run_{run_id:02d}/train_shot_{shot}.csv"
        val_csv = f"{data_root}/val.csv"
        if not os.path.exists(train_csv):
            print(f"Warning: training file not found: {train_csv}")
            continue

        train_dataset = CSVImageDataset(train_csv, preprocess, transform=augment, class_prompts=class_prompts)
        val_dataset = CSVImageDataset(val_csv, preprocess, transform=None, class_prompts=class_prompts)
        shot_output_dir = os.path.join(run_parent_dir, f"shot{shot}_adapter")
        os.makedirs(shot_output_dir, exist_ok=True)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
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

        fresh_base_model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model = CONCHWithAdapters(
            fresh_base_model,
            adapter_dim=args.adapter_dim,
            insert_vision_layers=insert_vision_layers,
            insert_text_layers=insert_text_layers,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        class_names = train_dataset.class_names
        evaluation_callback = UnifiedEvaluationCallback(
            model=model,
            tokenizer=tokenizer,
            val_dataloader=val_dataloader,
            class_names=class_names,
            output_dir=shot_output_dir,
            similarity_calculator=similarity_calculator,
            run_id=run_id,
            shot=shot,
            adapter_config={"adapter_dim": args.adapter_dim},
            class_prompts=class_prompts,
            patience=args.patience,
        )

        trainable_params = []
        if model.vision_adapters:
            trainable_params.extend(model.vision_adapters.parameters())
        if model.text_adapters:
            trainable_params.extend(model.text_adapters.parameters())
        if hasattr(model.base_model, "logit_scale"):
            trainable_params.append(model.base_model.logit_scale)

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, count = 0.0, 0

            for batch in tqdm(train_dataloader, desc=f"epoch {epoch}"):
                optimizer.zero_grad()
                images = batch["images"].to(device)
                text_tokens = move_tokens_to_device(batch["text_tokens"], device)

                img_feats, txt_feats = model(images, text_tokens)
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * (img_feats @ txt_feats.T)
                logits_per_text = logit_scale * (txt_feats @ img_feats.T)

                batch_size = img_feats.size(0)
                contrastive_labels = torch.arange(batch_size, device=device)
                loss_i2t = F.cross_entropy(logits_per_image, contrastive_labels)
                loss_t2i = F.cross_entropy(logits_per_text, contrastive_labels)
                loss = (loss_i2t + loss_t2i) / 2

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.base_model.logit_scale.clamp_(0, math.log(100))

                batch_items = images.size(0)
                total_loss += loss.item() * batch_items
                count += batch_items

            avg_loss = total_loss / max(1, count)
            should_stop = evaluation_callback.on_epoch_end(epoch, device, avg_loss)
            if should_stop:
                break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CONCH adapters with unified evaluation and similarity gap tracking"
    )
    parser.add_argument("--run_id", type=int, required=True, help="Run ID (1-10)")
    parser.add_argument("--shots", type=str, default="1", help="Shots per class, comma-separated, e.g. '1,2,4'")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--output_dir", type=str, default="output_conch_adapter", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Adapter bottleneck dimension")
    parser.add_argument(
        "--insert_vision_layers",
        type=str,
        default="6,7,8,9,10,11",
        help="Vision adapter layers, comma-separated, e.g. '6,7,8,9,10,11'; leave empty to disable vision adapters",
    )
    parser.add_argument(
        "--insert_text_layers",
        type=str,
        default="",
        help="Text adapter layers, comma-separated, e.g. '6,7,8,9,10,11'; leave empty to disable text adapters",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vandy",
        choices=["vandy", "cornell"],
        help="Dataset name used to select built-in data_root and class prompts",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Root directory for dataset (overrides --dataset default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_conch_adapter(args)

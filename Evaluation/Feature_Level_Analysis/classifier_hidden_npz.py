#!/usr/bin/env python3
"""
Extract classifier hidden embeddings and save them as NPZ files.

Outputs:
  {output_root}/{model}_classifier/{dataset}/run{NN}_shot{NN}/
    images.npz
    texts.npz
    metadata.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from conch.open_clip_custom import create_model_from_pretrained


DATASET_CONFIGS = {
    "cornell": {
        "test_csv": "/Data3/Daniel/Data1/Glom_Patches_0207_train_val_test_cornell/test.csv",
        "class_prompts": {
            "Atubular Glomeruli": "A pathology image of atubular glomerulus",
            "Global Glomerulosclerosis": "A pathology image of globally sclerotic glomerulus",
            "Ischemic Glomeruli": "A pathology image of ischemic glomerulus",
            "Segmental Glomerulosclerosis": "A pathology image of segmentally sclerotic glomerulus",
            "Viable Glomeruli": "A pathology image of viable glomerulus",
        },
    },
    "vandy": {
        "test_csv": "/Data3/Daniel/Data1/Glom_Patches_0207_train_val_test_vandy/test.csv",
        "class_prompts": {
            "Normal glomeruli": "A pathology image of a normal glomeruli",
            "Obsolescent glomeruli": "A pathology image of an obsolescent glomeruli",
            "Solidified glomeruli": "A pathology image of a solidified glomeruli",
            "Disappearing glomeruli": "A pathology image of a disappearing glomeruli",
            "Non-glomerular": "A pathology image of non-glomerular tissue",
        },
    },
}

MODEL_CONFIGS = {
    "clip": {"base_model": "openai/clip-vit-base-patch16"},
    "plip": {"base_model": "vinid/plip"},
    "conch": {"base_model": "conch_ViT-B-16", "pretrained": "hf_hub:MahmoodLab/conch"},
}


class MLPBatchNormClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

    def get_hidden_embedding(self, x):
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        return x


def infer_classifier_config(state_dict):
    config = {
        "input_dim": 512,
        "hidden_dim": 512,
        "num_classes": 5,
        "dropout": 0.5,
    }

    if "classifier.0.weight" in state_dict:
        config["hidden_dim"] = state_dict["classifier.0.weight"].shape[0]
        config["input_dim"] = state_dict["classifier.0.weight"].shape[1]

    if "classifier.4.weight" in state_dict:
        config["num_classes"] = state_dict["classifier.4.weight"].shape[0]

    return config


class CLIPBackbone:
    def __init__(self, model_name, device):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device

    def encode_images(self, images):
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=images)


class CONCHBackbone:
    def __init__(self, device):
        cfg = MODEL_CONFIGS["conch"]
        self.model, self.preprocess = create_model_from_pretrained(
            cfg["base_model"], cfg["pretrained"], device=device
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device

    def encode_images(self, images):
        with torch.no_grad():
            return self.model.encode_image(images)


def discover_classifier_paths(weight_root, model):
    pattern = os.path.join(
        weight_root, f"{model}_classifier", "run*_*", "shot*_*", "best_model", "classifier.pth"
    )
    paths = sorted(glob.glob(pattern))
    results = []
    for path in paths:
        parts = path.split(os.sep)
        run_dir = parts[-4]
        shot_dir = parts[-3]
        match_run = re.search(r"run(\d+)", run_dir)
        match_shot = re.search(r"shot(\d+)", shot_dir)
        if match_run and match_shot:
            run_id = int(match_run.group(1))
            shot = int(match_shot.group(1))
            weight_path = os.path.dirname(path)
            results.append((run_id, shot, path, weight_path))
    return results


def parse_run_ids(run_ids_str):
    if not run_ids_str:
        return None
    parts = re.split(r"[,\s]+", run_ids_str.strip())
    run_ids = []
    for part in parts:
        if part:
            run_ids.append(int(part))
    return sorted(set(run_ids))


def load_csv_data(csv_path, class_to_idx):
    df = pd.read_csv(csv_path)
    data_samples = []
    for _, row in df.iterrows():
        if row["class_name"] in class_to_idx:
            data_samples.append((row["path"], class_to_idx[row["class_name"]]))
    return data_samples


def extract_hidden_embeddings(model_type, backbone, classifier, csv_path, class_to_idx, batch_size, device):
    data_samples = load_csv_data(csv_path, class_to_idx)
    all_embeddings, all_labels, valid_paths = [], [], []

    for start in range(0, len(data_samples), batch_size):
        batch = data_samples[start : start + batch_size]
        batch_images, batch_labels, batch_paths = [], [], []

        for img_path, label_idx in batch:
            try:
                image = Image.open(img_path).convert("RGB")
                if model_type in ("clip", "plip"):
                    tensor = backbone.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
                else:
                    tensor = backbone.preprocess(image)
                batch_images.append(tensor)
                batch_labels.append(label_idx)
                batch_paths.append(img_path)
            except Exception:
                continue

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                features = backbone.encode_images(batch_tensor)
                hidden = classifier.get_hidden_embedding(features)
            all_embeddings.append(hidden.cpu().numpy())
            all_labels.extend(batch_labels)
            valid_paths.extend(batch_paths)

    if all_embeddings:
        return np.vstack(all_embeddings), np.array(all_labels), np.array(valid_paths)
    return np.array([]), np.array([]), np.array([])


def extract_text_features_placeholder(class_prompts, class_to_idx, feature_dim):
    prompts = list(class_prompts.values())
    labels = [class_to_idx[class_name] for class_name in class_prompts.keys()]
    class_names = list(class_prompts.keys())
    text_features = np.zeros((len(prompts), feature_dim), dtype=np.float32)
    return text_features, np.array(labels), np.array(prompts), np.array(class_names)


def save_features_to_npz(image_features, image_labels, image_paths, text_features, text_labels, text_prompts, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "images.npz"),
        X=image_features.astype(np.float32),
        y=image_labels.astype(np.int64),
        paths=np.array(image_paths, dtype=object),
    )

    np.savez_compressed(
        os.path.join(output_dir, "texts.npz"),
        X=text_features.astype(np.float32),
        y=text_labels.astype(np.int64),
        prompts=text_prompts,
        class_names=class_names,
    )


def create_metadata_csv(run_id, shot, model, method, weight_path, test_csv_path, num_images, output_dir):
    metadata = {
        "run_id": [run_id],
        "shot": [shot],
        "model": [model],
        "method": [method],
        "weight_path": [weight_path],
        "test_csv_path": [test_csv_path],
        "num_images": [num_images],
        "feature_type": ["hidden_embedding"],
        "extraction_layer": ["Layer_3_Dropout_output"],
    }
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def process_model_dataset(model, dataset, device, batch_size, output_root, weight_root, run_ids=None):
    cfg = DATASET_CONFIGS[dataset]
    test_csv = cfg["test_csv"]
    class_prompts = cfg["class_prompts"]
    class_to_idx = {class_name: index for index, class_name in enumerate(class_prompts.keys())}

    if model in ("clip", "plip"):
        backbone = CLIPBackbone(MODEL_CONFIGS[model]["base_model"], device)
    else:
        backbone = CONCHBackbone(device)

    classifier_entries = discover_classifier_paths(weight_root, model)
    if not classifier_entries:
        return

    if run_ids:
        classifier_entries = [entry for entry in classifier_entries if entry[0] in run_ids]

    for run_id, shot, classifier_path, weight_path in classifier_entries:
        state_dict = torch.load(classifier_path, map_location="cpu")
        config = infer_classifier_config(state_dict)
        classifier = MLPBatchNormClassifier(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )
        classifier.load_state_dict(state_dict)
        classifier.to(device).eval()

        embeddings, labels, paths = extract_hidden_embeddings(
            model, backbone, classifier, test_csv, class_to_idx, batch_size, device
        )

        if len(embeddings) == 0:
            del classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        text_features, text_labels, text_prompts, class_names = extract_text_features_placeholder(
            class_prompts, class_to_idx, config["hidden_dim"]
        )

        output_dir = os.path.join(output_root, f"{model}_classifier", dataset, f"run{run_id:02d}_shot{shot:02d}")
        save_features_to_npz(
            embeddings,
            labels,
            paths,
            text_features,
            text_labels,
            text_prompts,
            class_names,
            output_dir,
        )
        create_metadata_csv(run_id, shot, model, "classifier", weight_path, test_csv, len(embeddings), output_dir)

        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract classifier hidden embeddings and save them as NPZ files.")
    parser.add_argument("--dataset", default="all", choices=["cornell", "vandy", "all"])
    parser.add_argument("--model", default="all", choices=["clip", "plip", "conch", "all"])
    parser.add_argument("--output_root", default="/Data3/Daniel/fewshot/result_0207")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--run_ids", default="", help="Comma-separated run ids, for example: 8,9,10")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    datasets = ["cornell", "vandy"] if args.dataset == "all" else [args.dataset]
    models = ["clip", "plip", "conch"] if args.model == "all" else [args.model]
    run_ids = parse_run_ids(args.run_ids)

    for dataset in datasets:
        weight_root = f"/Data3/Daniel/fewshot/model_weight_0207_{dataset}"
        for model in models:
            process_model_dataset(model, dataset, device, args.batch_size, args.output_root, weight_root, run_ids=run_ids)


if __name__ == "__main__":
    main()

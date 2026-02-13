import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import random
from collections import defaultdict
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from src.datasets import ToyClsDataset
from src.model import TransformerLM


def set_random_seed(seed: int, cudnn_deterministic: bool = False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, _ = model(input_ids)
            logits = logits[:, -1, :]  # Get CLS token logits
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train(model, train_loader, eval_loader, optimizer, num_epochs, device, track_samples=None):
    """Train with optional attention tracking on fixed samples (one per label)."""
    # attn_history: dict mapping label -> list of (step, attn_all_layers_heads)
    # Each attn_all_layers_heads is a list of layers, each (num_heads, seq_len) for CLS row
    attn_history = {label: [] for label in track_samples.keys()} if track_samples else {}
    step = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Track attention on fixed samples
            if track_samples is not None:
                model.eval()
                with torch.no_grad():
                    for label, sample in track_samples.items():
                        _, all_attn_weights = model(sample.unsqueeze(0).to(device))
                        # all_attn_weights: list of (1, num_heads, seq, seq) per layer
                        # Extract CLS row (last row) for each layer and head
                        layer_head_attn = []
                        for layer_attn in all_attn_weights:
                            # layer_attn: (1, num_heads, seq, seq) -> (num_heads, seq) for CLS
                            cls_attn = layer_attn[0, :, -1, :].cpu().numpy()  # (num_heads, seq)
                            layer_head_attn.append(cls_attn.copy())
                        attn_history[label].append((step, layer_head_attn))
                model.train()
            
            step += 1

        train_loss = total_loss / len(train_loader)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
        if wandb.run is not None:
            wandb.log({
                "train/loss": train_loss,
                "eval/loss": eval_loss,
                "eval/acc": eval_acc,
            })
    
    return attn_history


@torch.no_grad()
def get_metrics_for_batch(model, input_ids, device):
    input_ids = input_ids.to(device)
    logits, all_attn_weights = model(input_ids)

    metrics = {}

    for layer_idx in range(model.num_layers):
        for head_idx in range(model.num_heads):
            attn_seq = all_attn_weights[layer_idx][:, head_idx, -1, :].cpu().numpy()
            attn_max = attn_seq.max()
            attn_sum = attn_seq.sum()
            metrics.update({
                f"{layer_idx}_{head_idx}_abs_attn_max": attn_max,
                f"{layer_idx}_{head_idx}_abs_attn_sum": attn_sum,
                f"{layer_idx}_{head_idx}_rel_attn_max": attn_max / attn_sum,
            })

    return metrics


def get_metrics_for_dataset(model, dataset, device):
    metrics = defaultdict(float)
    for input_ids, targets in dataset:
        batch_metrics = get_metrics_for_batch(model, input_ids.unsqueeze(0), device)
        for key, value in batch_metrics.items():
            metrics[key] += value
    for key, value in metrics.items():
        metrics[key] /= len(dataset)
    return metrics


CLS_TOKEN_ID = 0


@torch.no_grad()
def run_label_mixing_experiment(model, dataset, device, num_pairs=100, layer_idx=-1, head_idx=0):
    """
    Run minimal label mixing experiment.

    For each pair of samples from different labels:
    1. Get attention rankings (which tokens the CLS attends to most)
    2. Create two mixed samples:
       - Mixed A: top-1 token from label_a + all non-top-1 tokens from label_b
       - Mixed B: top-1 token from label_b + all non-top-1 tokens from label_a
    3. Check if model predicts top-1 label or the other label

    Returns metrics about prediction accuracy for mixed samples.

    Note: Dataset format is [tok1, tok2, ..., tokN, CLS]
    """
    model.eval()

    # Group samples by label
    samples_by_label = defaultdict(list)
    for i in range(len(dataset)):
        input_ids, label = dataset[i]
        samples_by_label[int(label)].append(input_ids)

    labels = sorted(samples_by_label.keys())
    if len(labels) < 2:
        return {"error": "need at least 2 labels"}

    def get_attention_ranking(input_ids):
        """Get content token indices sorted by CLS attention (descending).

        Returns indices into the content portion (positions 0 to -2), not full sequence.
        """
        input_ids_batch = input_ids.unsqueeze(0).to(device)
        _, all_attn_weights = model(input_ids_batch)
        # Use specified layer and head, CLS row (last row)
        cls_attn = all_attn_weights[layer_idx][0, head_idx, -1, :].cpu().numpy()
        # Only rank content tokens (exclude CLS at -1)
        content_attn = cls_attn[:-1]
        sorted_indices = np.argsort(content_attn)[::-1]
        return sorted_indices, content_attn

    # Track results
    results = {
        "mixed_a_predicts_top1_label": 0,  # Mixed A (top-1 from A) predicts label A
        "mixed_b_predicts_top1_label": 0,  # Mixed B (top-1 from B) predicts label B
        "total_pairs": 0,
    }

    # Run experiment on pairs of samples from different labels
    for _ in range(num_pairs):
        # Pick two random labels
        label_a, label_b = random.sample(labels, 2)

        # Pick random samples from each label
        sample_a = random.choice(samples_by_label[label_a])
        sample_b = random.choice(samples_by_label[label_b])

        # Extract content tokens (exclude CLS at -1)
        content_tokens_a = sample_a[:-1].tolist()
        content_tokens_b = sample_b[:-1].tolist()

        # Get attention rankings (indices into content tokens)
        ranking_a, _ = get_attention_ranking(sample_a)
        ranking_b, _ = get_attention_ranking(sample_b)

        # Create mixed samples (top-1 from one + rest from another)
        # Mixed A: top-1 from label_a + non-top-1 from label_b
        top1_idx_a = ranking_a[0]
        rest_indices_b = ranking_b[1:]  # all except top-1
        mixed_a_content = [content_tokens_a[top1_idx_a]] + [content_tokens_b[i] for i in rest_indices_b]
        random.shuffle(mixed_a_content)
        mixed_a = torch.tensor(mixed_a_content + [CLS_TOKEN_ID], dtype=torch.long)

        # Mixed B: top-1 from label_b + non-top-1 from label_a
        top1_idx_b = ranking_b[0]
        rest_indices_a = ranking_a[1:]  # all except top-1
        mixed_b_content = [content_tokens_b[top1_idx_b]] + [content_tokens_a[i] for i in rest_indices_a]
        random.shuffle(mixed_b_content)
        mixed_b = torch.tensor(mixed_b_content + [CLS_TOKEN_ID], dtype=torch.long)

        # Get predictions
        logits_a, _ = model(mixed_a.unsqueeze(0).to(device))
        logits_b, _ = model(mixed_b.unsqueeze(0).to(device))

        pred_a = logits_a[0, -1, :].argmax(dim=-1).item()  # Get CLS token logits
        pred_b = logits_b[0, -1, :].argmax(dim=-1).item()  # Get CLS token logits

        # Check if predictions match top-1 token's label
        if pred_a == label_a:
            results["mixed_a_predicts_top1_label"] += 1
        if pred_b == label_b:
            results["mixed_b_predicts_top1_label"] += 1

        results["total_pairs"] += 1

    # Compute rates
    total = results["total_pairs"]
    results["top1_dominance_rate_a"] = results["mixed_a_predicts_top1_label"] / total
    results["top1_dominance_rate_b"] = results["mixed_b_predicts_top1_label"] / total
    results["avg_top1_dominance_rate"] = (
        results["top1_dominance_rate_a"] + results["top1_dominance_rate_b"]
    ) / 2

    return results


@hydra.main(config_path="conf", config_name="classification_config", version_base=None)
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logging.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(project="softmax", name=cfg.wandb_name)
        wandb.run.mark_preempting()
        wandb.config.update(cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Attention type: {cfg.attention_type}")
    print(f"Positional type: {cfg.positional_type}")
    print(f"Vocab size: {cfg.vocab_size}")
    print(f"Sequence length: {cfg.seq_length}")

    # Create model using src version
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        output_dim=cfg.num_labels,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        use_mlp=cfg.use_mlp,
        use_layernorm=cfg.use_layernorm,
        attention_type=cfg.attention_type,
        is_causal=cfg.is_causal,
        positional_type=cfg.positional_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Config: {cfg.num_layers} layers, {cfg.num_heads} heads, {cfg.embed_dim} dim")

    # Create datasets and dataloaders for src experiment
    train_dataset = ToyClsDataset(cfg.num_train, cfg.seq_length, cfg.vocab_size, cfg.num_labels)
    eval_dataset = ToyClsDataset(cfg.num_eval, cfg.seq_length, cfg.vocab_size, cfg.num_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Sequence length: {train_dataset.input_ids[0].shape[0]}")

    # Train model using the imported train function
    train(model, train_loader, eval_loader, optimizer, cfg.num_epochs, device)

    metrics = get_metrics_for_dataset(model, eval_dataset, device)
    print(f"Metrics: {metrics}")

    if wandb.run is not None:
        wandb.log(metrics)

    # Run label mixing experiment
    for layer_idx in range(model.num_layers):
        for head_idx in range(model.num_heads):
            mixing_results = run_label_mixing_experiment(
                model, eval_dataset, device, num_pairs=500, layer_idx=layer_idx, head_idx=head_idx
            )
            print(f"Label mixing experiment results for layer {layer_idx} and head {head_idx}: {mixing_results}")

            if wandb.run is not None:
                wandb.log({f"mixing/{layer_idx}_{head_idx}/{k}": v for k, v in mixing_results.items()})

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

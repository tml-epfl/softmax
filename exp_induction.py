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

from src.datasets import InductionDataset
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


def evaluate(model, loader, device, aux_loss_weight):
    """Evaluate model on language modeling and induction accuracy."""
    model.eval()
    total_loss = 0
    correct_all = 0
    correct_induction = 0
    total_all = 0
    total_induction = 0
    
    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            logits, _ = model(input_ids)  # (batch, seq_len, vocab_size)

            logits_start = logits[:, :-1, :]
            logits_main = logits[:, -1, :]
            targets_start = targets[:, :-1]
            targets_main = targets[:, -1]

            loss_start = F.cross_entropy(logits_start.transpose(1, 2), targets_start)
            loss_main = F.cross_entropy(logits_main, targets_main)
            loss = (loss_start * aux_loss_weight + loss_main)
            total_loss += loss.item()

            # Overall accuracy (all positions)
            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            correct_all += (preds == targets).sum().item()
            total_all += targets.numel()

            induction_preds = preds[:, -1]  # (batch,)
            correct_induction += (induction_preds == targets_main).sum().item()
            total_induction += targets_main.size(0)
    
    avg_loss = total_loss / len(loader)
    acc_all = correct_all / total_all
    acc_induction = correct_induction / total_induction
    
    return avg_loss, acc_all, acc_induction


def train(model, train_loader, eval_loader, optimizer, num_epochs, device, aux_loss_weight):
    step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for input_ids, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids)  # (batch, seq_len, vocab_size)
            
            logits_start = logits[:, :-1, :]
            logits_main = logits[:, -1, :]
            targets_start = targets[:, :-1]
            targets_main = targets[:, -1]

            loss_start = F.cross_entropy(logits_start.transpose(1, 2), targets_start)
            loss_main = F.cross_entropy(logits_main, targets_main)
            loss = (loss_start * aux_loss_weight + loss_main)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            step += 1
        
        train_loss = total_loss / len(train_loader)
        eval_loss, eval_acc_all, eval_acc_induction = evaluate(model, eval_loader, device, aux_loss_weight)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Acc (all): {eval_acc_all:.4f} | Acc (induction): {eval_acc_induction:.4f}")
        if wandb.run is not None:
            wandb.log({
                "train/loss": train_loss,
                "eval/loss": eval_loss,
                "eval/acc_all": eval_acc_all,
                "eval/acc_induction": eval_acc_induction,
            })


def show_attention_maps(model, samples, device):
    for sample_id, (sample, _) in enumerate(samples):
        tokens = sample.tolist()
        input_seq = sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, all_attn_weights = model(input_seq)

        token_labels = ["BOS" if t == 0 else str(t) for t in tokens]

        # Create heatmap grid
        fig, axes = plt.subplots(model.num_layers, model.num_heads, figsize=(6 * model.num_heads, 5 * model.num_layers), squeeze=False)
        
        for layer_idx in range(model.num_layers):
            for head_idx in range(model.num_heads):
                ax = axes[layer_idx, head_idx]
                
                attn_matrix = all_attn_weights[layer_idx][0, head_idx].cpu().numpy()

                im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(token_labels)))
                ax.set_yticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(token_labels, fontsize=8)
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
                ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(f"Attention Heatmap: {sample_id}", fontsize=12)
        plt.tight_layout()
        
        if wandb.run is not None:
            wandb.log({f"attention_map_{sample_id}": wandb.Image(fig)})
        else:
            plt.show()


@torch.no_grad()
def get_metrics_for_batch(model, input_ids, device):
    input_ids = input_ids.to(device)
    logits, all_attn_weights = model(input_ids)

    metrics = {}

    for layer_idx in range(model.num_layers):
        for head_idx in range(model.num_heads):
            attn_matrix = all_attn_weights[layer_idx][:, head_idx].cpu().numpy()
            attn_bos = attn_matrix[:, 1 : -2, 0]
            attn_all = attn_matrix[:, 1 : -2, :].sum(axis=-1)
            metrics.update({
                f"{layer_idx}_{head_idx}_abs_attn_bos": attn_bos.mean(),
                f"{layer_idx}_{head_idx}_abs_attn_all": attn_all.mean(),
                f"{layer_idx}_{head_idx}_rel_attn_bos": (attn_bos / (attn_all + 1e-9)).mean(),
                f"{layer_idx}_{head_idx}_rel_attn_bos_global": attn_bos.sum() / (attn_all.sum() + 1e-9),
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


@hydra.main(config_path="conf", config_name="induction_config", version_base=None)
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
        vocab_size=cfg.vocab_size + int(cfg.shift_query_token) * cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        output_dim=cfg.vocab_size + int(cfg.shift_query_token) * cfg.vocab_size,
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
    train_dataset = InductionDataset(cfg.num_train, cfg.vocab_size, cfg.seq_length, shift_query_token=cfg.shift_query_token)
    eval_dataset = InductionDataset(cfg.num_eval, cfg.vocab_size, cfg.seq_length, shift_query_token=cfg.shift_query_token)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Sequence length: {train_dataset.sequences[0].shape[0]}")

    # Train model using the imported train function
    train(model, train_loader, eval_loader, optimizer, cfg.num_epochs, device, cfg.aux_loss_weight)

    # Visualize attention maps for some samples
    samples_for_viz = [eval_dataset[i] for i in range(2)]
    show_attention_maps(model, samples_for_viz, device)

    metrics = get_metrics_for_dataset(model, eval_dataset, device)
    print(f"Metrics: {metrics}")

    if wandb.run is not None:
        wandb.log(metrics)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

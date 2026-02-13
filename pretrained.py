# General imports.
import argparse
import random
from typing import List, Literal, Sequence

# JAX must be imported before tensorflow.
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import os
import subprocess
from pathlib import Path
from datasets import load_dataset
import matplotlib.pyplot as plt

# Necessary so that the checkpoint loader works inside a notebook.
import nest_asyncio
import numpy as np
nest_asyncio.apply()

# The LM configurations and checkpoints.
from axlearn.common.config import config_for_function, get_named_trainer_config

# Stuff to configure the local device mesh.
from axlearn.common import utils_spmd

# To control what to load from the saved checkpoint.
from axlearn.common import state_builder
from axlearn.common.checkpointer import CheckpointValidationType

# For the tokenizer/vocab.
from axlearn.experiments.text import common

# For typing stuff.
from axlearn.common.utils import DataPartitionType, set_data_dir
from axlearn.common.inference import InferenceRunner

from tqdm import trange


JAX_BACKEND: Literal["cpu", "tpu", "gpu"] = "cpu"
DATA_DIR: str = "gs://axlearn-public/tensorflow_datasets"

REMOTE_MODEL_DIR = "gs://axlearn-public/experiments/"
LOCAL_MODEL_DIR = str(Path(__file__).resolve().parent / "data/")
MODEL_INFO: dict[str, dict[str, str]] = {
    # Sigmoid-based attention.
    "7b-sigmoid": {
        "checkpoint_dir": "gala-7B-sigmoid-hybridnorm-alibi-sprp-2024-12-03-1002/checkpoints/step_00250000",
        "config_name": "gala-sigmoid-7B-4k-hybridnorm-alibi-sp-rp",
        "sentencepiece_model_name": "bpe_32k_c4.model",
        "config_module": "axlearn.experiments.text.gpt.pajama_sigmoid_trainer",
    },
    # Softmax baseline.
    "7b-softmax": {
        "checkpoint_dir": "gala-7B-hybridnorm-alibi-sprp-2024-12-02-1445/checkpoints/step_00250000",
        "config_name": "gala-7B-hybridnorm-alibi-flash-sp-rp",
        "sentencepiece_model_name": "bpe_32k_c4.model",
        "config_module": "axlearn.experiments.text.gpt.pajama_trainer",
    },
}

utils_spmd.setup(jax_backend=JAX_BACKEND)


def _init_state_builder_discard_optimizer(
    *,
    source_config_name: str,
    source_config_module: str,
    mesh_axis_names: Sequence[str],
    mesh_shape: Sequence[int],
    checkpoint_dir: str,
) -> state_builder.Builder.Config:
    converter = state_builder.ModelStateScopeConverter.default_config().set(
        source_trainer_config=config_for_function(get_named_trainer_config).set(
            config_name=source_config_name,
            config_module=source_config_module,
        ),
        # Only keep `decoder` tree, which means we throw away optimizer.
        scope={"decoder": "decoder"},
        mesh_axis_names=mesh_axis_names,
        mesh_shape=mesh_shape,
    )
    init_state_builder = state_builder.RestoreAndConvertBuilder.default_config().set(
        builder=state_builder.TensorStoreStateStorageBuilder.default_config().set(
            validation=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE,
            dir=checkpoint_dir,
        ),
        converter=converter,
    )
    return init_state_builder


def get_inference_runner(name: str, param_dtype: jnp.dtype) -> InferenceRunner:
    """Make an inference runner initialized with pre-trained state according to model name."""
    ckpt_dir = MODEL_INFO[name]["checkpoint_dir"]

    # If we don't have a local version, first download it.
    local_ckpt_dir = Path(LOCAL_MODEL_DIR) / ckpt_dir
    if not local_ckpt_dir.exists():
        remote_ckpt_dir = os.path.join(REMOTE_MODEL_DIR, ckpt_dir)
        print(f"Copying checkpoint from {remote_ckpt_dir} to {local_ckpt_dir}.")
        os.makedirs(local_ckpt_dir, exist_ok=True)
        os.makedirs(local_ckpt_dir / "gda", exist_ok=True)
        print(f"Copying checkpoint from {remote_ckpt_dir} to {local_ckpt_dir}.")
        subprocess.run(["gsutil", "-m", "cp", "-r", os.path.join(remote_ckpt_dir, "tf_*"), local_ckpt_dir])
        subprocess.run(["gsutil", "-m", "cp", "-r", os.path.join(remote_ckpt_dir, "gda", "model"), local_ckpt_dir / "gda"])
        subprocess.run(["gsutil", "-m", "cp", "-r", os.path.join(remote_ckpt_dir, "gda", "prng_key"), local_ckpt_dir / "gda"])
        subprocess.run(["gsutil", "cp", os.path.join(remote_ckpt_dir, "index"), local_ckpt_dir])

    config_name = MODEL_INFO[name]["config_name"]
    config_module = MODEL_INFO[name]["config_module"]
    mesh_axis_names = (
        "data",
        "expert",
        "fsdp",
        "model",
        "seq",
    )
    mesh_shape = (
        1,
        1,
        1,
        len(jax.devices()),
        1,
    )

    trainer_cfg = get_named_trainer_config(
        config_name=config_name,
        config_module=config_module,
    )()

    init_state_builder = _init_state_builder_discard_optimizer(
        source_config_name=config_name,
        source_config_module=config_module,
        mesh_axis_names=mesh_axis_names,
        mesh_shape=mesh_shape,
        checkpoint_dir=str(local_ckpt_dir),
    )

    inference_runner_cfg = InferenceRunner.default_config().set(
        name=f"{name}_inference_runner",
        mesh_axis_names=mesh_axis_names,
        mesh_shape=mesh_shape,
        model=trainer_cfg.model.set(dtype=param_dtype),
        input_batch_partition_spec=DataPartitionType.REPLICATED,
        init_state_builder=init_state_builder,
    )
    print(f"Loading state for {name} from:\n{local_ckpt_dir}")
    inference_runner = inference_runner_cfg.instantiate(parent=None)
    return inference_runner


def get_vocab(name: str) -> seqio.Vocabulary:
    """Get the vocabulary based on the model's name."""
    with set_data_dir(DATA_DIR):
        vocab = common.vocab(
            sentencepiece_model_name=MODEL_INFO[name]["sentencepiece_model_name"]
        )
    return vocab


def _preprocess_text(text: str) -> str:
    """Preprocesses text for tokenization."""
    return text.replace("\n", "<n>")


def compute_attention_stats_detailed_2(
    prompts: List[str],
    inference_runner: InferenceRunner,
    vocab: seqio.Vocabulary,
    max_seq_len: int = 256,
    batch_size: int = 4,
) -> dict:
    """Compute detailed attention statistics including full weight distribution.

    Args:
        prompts: List of text prompts to analyze.
        inference_runner: The initialized inference runner.
        vocab: The vocabulary for tokenization.
        max_seq_len: Maximum sequence length for tokenization.
        batch_size: Number of prompts to process at once.

    Returns:
        Dict containing:
        - max_attention_weight: Average proportion of weight on largest attention weight
        - proportion_below_threshold: Average proportion of weights < 0.001
        - proportion_above_threshold: Average proportion of weights > 0.1
        - all_weights: Flattened numpy array of all attention weights for plotting
    """
    # Disable FlashAttention to get real attention probs (FlashAttention returns empty probs).
    import axlearn.common.flash_attention.layer as flash_layer
    original_flash_impl = flash_layer.flash_attention_implementation
    flash_layer.flash_attention_implementation = lambda **kwargs: None

    # Create method runner that captures module outputs including full probs
    drop_fn = lambda path: False  # Don't drop anything
    method_runner = inference_runner.create_method_runner(
        method="predict",
        prng_key=jax.random.PRNGKey(42),
        drop_module_outputs=drop_fn,
        return_aux={"self_attention_probs"},  # Request full attention probs
    )

    # Accumulate statistics across batches
    max_proportion = np.zeros((32, 32))
    first_token_proportion = np.zeros((32, 32))
    number_of_tokens = 0

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in trange(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Tokenize batch
        input_ids_list = []
        for prompt in batch_prompts:
            tokens = vocab.encode(_preprocess_text(prompt))
            # Pad or truncate to max_seq_len
            if len(tokens) >= max_seq_len:
                tokens = tokens[:max_seq_len]
            else:
                print(len(tokens))
                tokens = tokens + [vocab.pad_id] * (max_seq_len - len(tokens))
            input_ids_list.append(tokens)

        input_ids_np = np.asarray(input_ids_list, dtype=np.int32)           # [B, T]
        valid_q_np = (input_ids_np != vocab.pad_id)                         # [B, T]
        mask = jnp.asarray(valid_q_np)[None, :, None, :] 

        input_batch = {"input_ids": jnp.asarray(input_ids_list, dtype=jnp.int32)}

        # Run forward pass
        runner_output = method_runner(input_batch)

        # Extract attention metrics
        module_outputs = runner_output.module_outputs
        attention_outputs = module_outputs["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]
        max_proportion += np.array((attention_outputs["max_attention_weight"] * mask).sum(axis=(1, 3)))
        first_token_proportion += np.array((attention_outputs["first_token_proportion_of_total_weight"] * mask).sum(axis=(1, 3)))
        number_of_tokens += int(valid_q_np.sum())

    # Restore original flash attention implementation
    flash_layer.flash_attention_implementation = original_flash_impl

    return {
        "max_proportion": max_proportion / number_of_tokens,
        "first_token_proportion": first_token_proportion / number_of_tokens,
    }




def compute_attention_stats_detailed(
    prompts: List[str],
    inference_runner: InferenceRunner,
    vocab: seqio.Vocabulary,
    max_seq_len: int = 256,
    batch_size: int = 4,
) -> dict:
    """Compute detailed attention statistics including full weight distribution.

    Args:
        prompts: List of text prompts to analyze.
        inference_runner: The initialized inference runner.
        vocab: The vocabulary for tokenization.
        max_seq_len: Maximum sequence length for tokenization.
        batch_size: Number of prompts to process at once.

    Returns:
        Dict containing:
        - max_attention_weight: Average proportion of weight on largest attention weight
        - proportion_below_threshold: Average proportion of weights < 0.001
        - proportion_above_threshold: Average proportion of weights > 0.1
        - all_weights: Flattened numpy array of all attention weights for plotting
    """
    # Disable FlashAttention to get real attention probs (FlashAttention returns empty probs).
    import axlearn.common.flash_attention.layer as flash_layer
    original_flash_impl = flash_layer.flash_attention_implementation
    flash_layer.flash_attention_implementation = lambda **kwargs: None

    # Create method runner that captures module outputs including full probs
    drop_fn = lambda path: False  # Don't drop anything
    method_runner = inference_runner.create_method_runner(
        method="predict",
        prng_key=jax.random.PRNGKey(42),
        drop_module_outputs=drop_fn,
        return_aux={"self_attention_probs"},  # Request full attention probs
    )

    # Accumulate statistics across batches
    all_max_attn = []
    all_prop_below = []
    all_prop_above = []

    all_prop_of_total_weight = []
    all_first_token_prop_of_total_weight = []
    all_first_token_attn_weight = []
    all_weights_samples = []  # Store sampled weights for plotting
    max_flattened = []

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in trange(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Tokenize batch
        input_ids_list = []
        for prompt in batch_prompts:
            tokens = vocab.encode(_preprocess_text(prompt))
            # Pad or truncate to max_seq_len
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            else:
                tokens = tokens + [vocab.pad_id] * (max_seq_len - len(tokens))
            input_ids_list.append(tokens)

        input_batch = {"input_ids": jnp.asarray(input_ids_list, dtype=jnp.int32)}

        # Run forward pass
        runner_output = method_runner(input_batch)

        # Extract attention metrics
        module_outputs = runner_output.module_outputs
        attention_outputs = module_outputs["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]

        all_max_attn.append(float(jnp.mean(attention_outputs["max_attention_weight"])))
        all_prop_below.append(float(jnp.mean(attention_outputs["proportion_below_threshold"])))
        all_prop_above.append(float(jnp.mean(attention_outputs["proportion_above_threshold"])))
        all_prop_of_total_weight.append(float(jnp.mean(attention_outputs["proportion_of_total_weight"])))
        all_first_token_prop_of_total_weight.append(float(jnp.mean(attention_outputs["first_token_proportion_of_total_weight"])))
        all_first_token_attn_weight.append(float(jnp.mean(attention_outputs["first_token_attention_weight"])))
        max_flattened.append(np.array(attention_outputs["max_attention_weight"]).flatten())
        # Get full attention probs and sample for plotting
        # probs shape: [num_layers, batch, num_heads, target_length, source_length]
        probs = module_outputs["decoder"]["transformer"]["repeat"]["layer"]["self_attention_probs"]
        probs_flat = np.array(probs).flatten()

        # Sample to avoid memory issues (sample up to 100k weights per batch)
        max_samples = 100000
        if len(probs_flat) > max_samples:
            sample_indices = np.random.choice(len(probs_flat), max_samples, replace=False)
            probs_flat = probs_flat[sample_indices]
        all_weights_samples.append(probs_flat)

        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"max_attn={all_max_attn[-1]:.4f}, "
              f"below={all_prop_below[-1]:.4f}, "
              f"above={all_prop_above[-1]:.4f}")

    # Restore original flash attention implementation
    flash_layer.flash_attention_implementation = original_flash_impl

    # Combine all sampled weights
    all_weights = np.concatenate(all_weights_samples)

    return {
        "max_attention_weight": sum(all_max_attn) / len(all_max_attn),
        "proportion_below_threshold": sum(all_prop_below) / len(all_prop_below),
        "proportion_above_threshold": sum(all_prop_above) / len(all_prop_above),
        "proportion_of_total_weight": sum(all_prop_of_total_weight) / len(all_prop_of_total_weight),
        "first_token_proportion_of_total_weight": sum(all_first_token_prop_of_total_weight) / len(all_first_token_prop_of_total_weight),
        "first_token_attention_weight": sum(all_first_token_attn_weight) / len(all_first_token_attn_weight),
        # "all_weights": all_weights,
        # "max_flattened": np.concatenate(max_flattened),
    }


def load_pile_samples(n_samples: int, seed: int = 42) -> List[str]:
    """Load random samples from NeelNanda/pile-10k dataset."""
    print(f"Loading NeelNanda/pile-10k dataset...")
    dataset = load_dataset("NeelNanda/pile-10k", split="train")

    # Sample random indices
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    prompts = [dataset[i]["text"] for i in indices]
    print(f"Loaded {len(prompts)} samples from pile-10k")
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute detailed attention statistics with distribution plots")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sigmoid", "softmax"],
        default="sigmoid",
        help="Model type: sigmoid or softmax"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to use from pile-10k (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Output directory"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model = args.model
    results = {}

    model_name = f"7b-{model}"
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    inference_runner = get_inference_runner(model_name, param_dtype=jnp.bfloat16)
    vocab = get_vocab(model_name)

    # Load samples from pile-10k
    prompts = load_pile_samples(args.n_samples, seed=args.seed)

    print(f"\nComputing attention stats for {len(prompts)} prompts (max_seq_len={args.max_seq_len}, batch_size={args.batch_size})...")
    stats = compute_attention_stats_detailed_2(
        prompts, inference_runner, vocab,
        max_seq_len=args.max_seq_len, batch_size=args.batch_size
    )
    results[model] = stats

    print(results)

    np.save(os.path.join(args.output_dir, f"attention_stats_detailed_{model}.npy"), results)

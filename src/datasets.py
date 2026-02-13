import torch
import torch.utils.data as data
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset


BOS_TOKEN_ID = 0
CLS_TOKEN_ID = 0


class InductionDataset(Dataset):
    """Dataset for causal language modeling with induction pattern."""

    def __init__(
        self,
        num_samples: int,
        vocab_size: int = 50,
        seq_length: int = 12,
        shift_query_token: bool = False,
    ):
        """
        Args:
            num_samples: Number of sequences to generate
            vocab_size: Total vocabulary size (including BOS)
            seq_length: Total sequence length (including BOS and repeated pair)
        """
        self.shift_query_token = shift_query_token
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        # Available tokens (excluding BOS)
        self.available_tokens = list(range(1, vocab_size))
        
        # Generate sequences
        self.sequences = []
        
        for _ in range(num_samples):
            seq = self._generate_sequence()
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
    
    def _generate_sequence(self) -> List[int]:
        """Generate a single sequence ending with a repeated pair.
        
        Returns:
            sequence: [BOS, token1, token2, ..., tokenN, tokenN]
                     where the last token repeats the second-to-last
        """
        # Generate random unique tokens for the main sequence
        # We need seq_length - 3 tokens (excluding BOS and the final repeated pair)
        num_main_tokens = self.seq_length - 3
        
        # Sample unique tokens
        main_tokens = random.sample(self.available_tokens, num_main_tokens)
        
        # Choose a random position to create the induction pattern
        # The token at this position will be repeated at the end
        repeat_idx = random.randint(0, num_main_tokens - 1)
        repeated_token = main_tokens[repeat_idx]
        if self.shift_query_token:
            repeated_token += self.vocab_size
        continuation = (main_tokens + [repeated_token])[repeat_idx + 1]
        
        # Build sequence: [BOS, main_tokens..., repeated_token]
        sequence = [BOS_TOKEN_ID] + main_tokens + [repeated_token, continuation]

        return sequence

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return input and target for language modeling.
        
        Returns:
            input_ids: sequence[:-1]  (all but last token)
            targets: sequence[1:]     (all but first token)
        """
        seq = self.sequences[idx]
        input_ids = seq[:-1]   # [BOS, tok1, tok2, ..., tokN, repeated_tok]
        targets = seq[1:]      # [tok1, tok2, ..., tokN, repeated_tok, continuation]
        return input_ids, targets


def print_induction_sequence(seq_tensor):
    """Pretty print an induction sequence."""
    seq_list = seq_tensor.tolist()

    def token2str(t):
        if t == 0:
            return "BOS"
        else:
            return str(t)

    return " ".join([token2str(t) for t in seq_list])


class ToyClsDataset(Dataset):
    """Token group classification dataset with CLS token appended."""

    def __init__(
        self,
        num_samples: int,
        seq_length: int, # including CLS
        vocab_size: int, # including CLS
        num_labels: int,
    ):
        assert (vocab_size - 1) % num_labels == 0
        self.num_samples = num_samples
        self.seq_length = seq_length - 1
        self.vocab_size = vocab_size - 1
        self.num_labels = num_labels
        self.group_size = self.vocab_size // num_labels

        # Randomly assign each sample to a group (label)
        self.labels = torch.randint(0, num_labels, (num_samples,))

        # Generate tokens from the assigned group only (without replacement)
        # Sequence: [tok1, tok2, ..., tokN, CLS]
        self.input_ids = torch.zeros(num_samples, self.seq_length + 1, dtype=torch.long)
        for i in range(num_samples):
            group = self.labels[i].item()
            low = group * self.group_size
            perm = torch.randperm(self.group_size)[:self.seq_length] + low + 1  # +1 to reserve 0 for CLS
            self.input_ids[i, :self.seq_length] = perm
            self.input_ids[i, self.seq_length] = CLS_TOKEN_ID  # Append CLS at the end

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]

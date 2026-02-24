"""Trainer code for PCFG language models.

Trains a sentence-level (one sentence per line) GPT-2 style causal language model on
word-level tokens, with explicit '<bos>' and '<eos>' tokens added to each sentence.

Author: OpenAI Codex and Freda Shi <fhs@uwaterloo.ca>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence


@dataclass(frozen=True, slots=True)
class WordVocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_token: str = '<pad>'
    bos_token: str = '<bos>'
    eos_token: str = '<eos>'
    unk_token: str = '<unk>'

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]


def _resolve_path(path_str: str, *, script_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir / path


def _load_sentences(path: Path) -> List[List[str]]:
    sentences: List[List[str]] = []
    with path.open(encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            sentences.append(line.split())
    return sentences


def _build_vocab(
    sentences: Sequence[Sequence[str]],
    *,
    pad_token: str,
    bos_token: str,
    eos_token: str,
    unk_token: str,
) -> WordVocab:
    specials = [pad_token, bos_token, eos_token, unk_token]
    tokens = sorted({tok for sent in sentences for tok in sent if tok not in specials})
    id_to_token = specials + tokens
    token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}
    return WordVocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train a GPT2 LM on PCFG samples (word-level).')
    parser.add_argument(
        'data_file',
        help='Path to a samples file (e.g., samples/pcfg1.csv). Interpreted relative to this script unless absolute.',
    )
    parser.add_argument(
        '--output-dir',
        default='checkpoints',
        help='Directory to write model checkpoints to (relative to this script unless absolute).',
    )

    parser.add_argument('--pad-token', default='<pad>')
    parser.add_argument('--bos-token', default='<bos>')
    parser.add_argument('--eos-token', default='<eos>')
    parser.add_argument('--unk-token', default='<unk>')
    parser.add_argument('--max-seq-len', type=int, default=128)

    parser.add_argument('--n-embed', type=int, default=128, help='Hidden size (GPT-2 n_embd).')
    parser.add_argument('--n-head', type=int, default=4, help='Number of attention heads (GPT-2 n_head).')
    parser.add_argument('--n-layer', type=int, default=4, help='Number of Transformer layers (GPT-2 n_layer).')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    if args.max_seq_len < 2:
        raise SystemExit('--max-seq-len must be >= 2 (needs room for <bos> and <eos>)')
    if args.grad_accum_steps < 1:
        raise SystemExit('--grad-accum-steps must be >= 1')

    script_dir = Path(__file__).resolve().parent
    data_path = _resolve_path(args.data_file, script_dir=script_dir)
    output_dir = _resolve_path(args.output_dir, script_dir=script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    sentences = _load_sentences(data_path)
    if not sentences:
        raise SystemExit(f'No non-empty lines found in {data_path}')

    sentences_with_bounds: List[List[str]] = []
    for sent in sentences:
        truncated = list(sent)
        if len(truncated) > args.max_seq_len - 2:
            truncated = truncated[: args.max_seq_len - 2]
        bounded = [args.bos_token] + truncated + [args.eos_token]
        sentences_with_bounds.append(bounded)

    vocab = _build_vocab(
        sentences_with_bounds,
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        unk_token=args.unk_token,
    )

    encoded: List[List[int]] = [vocab.encode(sent) for sent in sentences_with_bounds]

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
    except ModuleNotFoundError as e:
        raise SystemExit(
            'Missing dependency. Please install PyTorch + HuggingFace Transformers.\n'
            'Example: pip install \'torch\' \'transformers\'\n'
            f'Original error: {e}'
        ) from e

    class SentenceDataset(Dataset):
        def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
            self._sequences = sequences

        def __len__(self) -> int:
            return len(self._sequences)

        def __getitem__(self, idx: int) -> List[int]:
            return list(self._sequences[idx])

    def collate_batch(batch: Sequence[Sequence[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(seq) for seq in batch)
        input_ids = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, seq in enumerate(batch):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    dataset = SentenceDataset(encoded)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = GPT2Config(
        vocab_size=len(vocab.id_to_token),
        n_positions=args.max_seq_len,
        n_ctx=args.max_seq_len,
        n_embd=args.n_embed,
        n_head=args.n_head,
        n_layer=args.n_layer,
        bos_token_id=vocab.bos_id,
        eos_token_id=vocab.eos_id,
        pad_token_id=vocab.pad_id,
    )
    model = GPT2LMHeadModel(config).to(device)

    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or 'ln_' in name or '.ln_' in name or 'layernorm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=args.learning_rate,
    )

    steps_per_epoch = len(dataloader)
    total_update_steps = (steps_per_epoch * args.epochs + args.grad_accum_steps - 1) // args.grad_accum_steps
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_update_steps, desc='train', dynamic_ncols=True)
    update_step = 0
    loss_ema: float | None = None  # exponential moving average of loss for smoother display

    for epoch in range(args.epochs):
        for step_in_epoch, batch in enumerate(dataloader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if step_in_epoch % args.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            update_step += 1
            loss_value = float(outputs.loss.detach().cpu().item())
            loss_ema = loss_value if loss_ema is None else (0.9 * loss_ema + 0.1 * loss_value)

            pbar.update(1)
            pbar.set_postfix(
                epoch=epoch + 1,
                loss=f'{loss_value:.4f}',
                loss_ema=f'{loss_ema:.4f}',
                lr=f'{scheduler.get_last_lr()[0]:.2e}',
            )

        if step_in_epoch % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            update_step += 1
            pbar.update(1)

    pbar.close()

    checkpoint_path = output_dir / f'{data_path.stem}.pt'
    torch.save(
        {
            'state_dict': model.state_dict(),
            'config': config.to_dict(),
            'vocab': vocab.token_to_id,
            'special_tokens': {
                'pad_token': vocab.pad_token,
                'bos_token': vocab.bos_token,
                'eos_token': vocab.eos_token,
                'unk_token': vocab.unk_token,
            },
        },
        checkpoint_path,
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Sampling code for PCFG language models.

This script demonstrates how to load a trained GPT-2 checkpoint, compute next-token probabilities, and sample sentences.
1) loading a trained model checkpoint
2) computing next-token probabilities
3) sampling sequences

Example call:
    python sampling.py checkpoints/pcfg1.pt --num-samples 5 --top-k 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]


def _resolve_path(path_str: str, *, script_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir / path


def _build_id_to_token(token_to_id: Dict[str, int]) -> List[str]:
    if not token_to_id:
        raise ValueError('Vocabulary is empty')

    vocab_size = max(token_to_id.values()) + 1
    id_to_token = [''] * vocab_size
    for token, token_id in token_to_id.items():
        if token_id < 0:
            raise ValueError(f'Found negative token id {token_id} for token {token!r}')
        id_to_token[token_id] = token

    missing = [idx for idx, token in enumerate(id_to_token) if token == '']
    if missing:
        raise ValueError(f'Vocabulary ids are not contiguous. Missing ids: {missing[:10]}')
    return id_to_token


def _load_checkpoint(checkpoint_path: Path, *, device: 'torch.device') -> Tuple['GPT2LMHeadModel', LoadedArtifacts]:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    payload = torch.load(checkpoint_path, map_location=device)
    required = {'state_dict', 'config', 'vocab'}
    if not required.issubset(payload):
        raise ValueError(f'Checkpoint missing required keys: {sorted(required - set(payload.keys()))}')

    token_to_id = payload['vocab']
    if not isinstance(token_to_id, dict):
        raise ValueError('Checkpoint key "vocab" must be a dict[token, id]')

    specials = payload.get('special_tokens', {})
    bos_token = specials.get('bos_token', '<bos>')
    eos_token = specials.get('eos_token', '<eos>')
    pad_token = specials.get('pad_token', '<pad>')
    unk_token = specials.get('unk_token', '<unk>')

    if bos_token not in token_to_id and eos_token in token_to_id:
        bos_token = eos_token

    for special in [bos_token, eos_token, pad_token, unk_token]:
        if special not in token_to_id:
            raise ValueError(f'Special token {special!r} is missing from vocabulary')

    config = GPT2Config(**payload['config'])
    model = GPT2LMHeadModel(config)
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()

    artifacts = LoadedArtifacts(
        token_to_id=token_to_id,
        id_to_token=_build_id_to_token(token_to_id),
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )
    return model, artifacts


def _encode_prompt(prompt: str, *, token_to_id: Dict[str, int], unk_id: int) -> List[int]:
    stripped = prompt.strip()
    if not stripped:
        return []
    return [token_to_id.get(token, unk_id) for token in stripped.split()]


def _decode_sentence(token_ids: Sequence[int], *, artifacts: LoadedArtifacts) -> str:
    tokens: List[str] = []
    for token_id in token_ids:
        if token_id == artifacts.eos_id:
            break
        if token_id == artifacts.pad_id:
            continue
        tokens.append(artifacts.id_to_token[token_id])
    return ' '.join(tokens)


def _next_token_probs(
    model: 'GPT2LMHeadModel',
    *,
    prefix_ids: Sequence[int],
    device: 'torch.device',
) -> 'torch.Tensor':
    import torch

    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    return probs


def _sample_next_token_id(
    probs: 'torch.Tensor',
    *,
    top_k: int,
) -> int:
    import torch

    if top_k > 0 and top_k < probs.size(-1):
        top_probs, top_indices = torch.topk(probs, k=top_k)
        renorm_probs = top_probs / top_probs.sum()
        picked = int(torch.multinomial(renorm_probs, num_samples=1).item())
        return int(top_indices[picked].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def _sample_sentence_ids(
    model: 'GPT2LMHeadModel',
    *,
    prefix_ids: Sequence[int],
    artifacts: LoadedArtifacts,
    top_k: int,
    max_new_tokens: int,
    device: 'torch.device',
) -> List[int]:
    generated = list(prefix_ids)
    for _ in range(max_new_tokens):
        probs = _next_token_probs(
            model,
            prefix_ids=generated,
            device=device,
        )
        next_token_id = _sample_next_token_id(probs, top_k=top_k)
        generated.append(next_token_id)
        if next_token_id == artifacts.eos_id:
            break
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Load a GPT-2 checkpoint, inspect next-token probabilities, and sample sentences.'
    )
    parser.add_argument(
        'checkpoint',
        help='Path to model checkpoint (e.g., checkpoints/pcfg1.pt). Interpreted relative to this script unless absolute.',
    )
    parser.add_argument('--num-samples', type=int, default=100, help='Number of sampled sentences.')
    parser.add_argument('--prompt', default='', help='Optional prompt (space-separated word tokens). <bos> is prepended automatically.')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling (0 disables top-k filtering).')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--show-top-k-probs', type=int, default=10, help='How many next-token probabilities to print.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument(
        '--output-file',
        default='',
        help='Optional path to save sampled sentences (one line each). Relative paths are resolved from this script directory.',
    )
    args = parser.parse_args(argv)

    if args.num_samples < 1:
        raise SystemExit('--num-samples must be >= 1')
    if args.top_k < 0:
        raise SystemExit('--top-k must be >= 0')
    if args.max_new_tokens < 1:
        raise SystemExit('--max-new-tokens must be >= 1')
    if args.show_top_k_probs < 1:
        raise SystemExit('--show-top-k-probs must be >= 1')

    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            'Missing dependency. Please install PyTorch + HuggingFace Transformers.\n'
            'Example: pip install \'torch\' \'transformers\'\n'
            f'Original error: {e}'
        ) from e

    script_dir = Path(__file__).resolve().parent
    checkpoint_path = _resolve_path(args.checkpoint, script_dir=script_dir)
    if not checkpoint_path.exists():
        raise SystemExit(f'Checkpoint not found: {checkpoint_path}')

    output_path: Path | None = None
    if args.output_file:
        output_path = _resolve_path(args.output_file, script_dir=script_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise SystemExit('CUDA requested but not available')
        device = torch.device(args.device)

    model, artifacts = _load_checkpoint(checkpoint_path, device=device)

    prompt_ids = [artifacts.bos_id]
    prompt_ids.extend(_encode_prompt(args.prompt, token_to_id=artifacts.token_to_id, unk_id=artifacts.unk_id))

    probs = _next_token_probs(model, prefix_ids=prompt_ids, device=device)
    top_k = min(args.show_top_k_probs, probs.size(-1))
    top_probs, top_indices = torch.topk(probs, k=top_k)

    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f'Device: {device}')
    print(f'Prompt: {args.prompt!r}')
    print(f'Top {top_k} next-token probabilities:')
    for idx in range(top_k):
        token_id = int(top_indices[idx].item())
        prob = float(top_probs[idx].item())
        token = artifacts.id_to_token[token_id]
        print(f'  {token}\t{prob:.6f}')
    print('')

    sampled_sentences: List[str] = []
    for sample_idx in range(args.num_samples):
        sentence_ids = _sample_sentence_ids(
            model,
            prefix_ids=prompt_ids,
            artifacts=artifacts,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        continuation_ids = sentence_ids[len(prompt_ids) :]
        sentence_text = _decode_sentence(continuation_ids, artifacts=artifacts)
        sampled_sentences.append(sentence_text)
        print(f'{sample_idx + 1:03d}: {sentence_text}')

    if output_path is not None:
        with output_path.open('w', encoding='utf-8') as f:
            for sentence in sampled_sentences:
                f.write(sentence + '\n')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

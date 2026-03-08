"""
task3_hybrid.py

Hybrid BPE + Inside-Outside EM for recovering PCFG 3.

Usage:
    python task3_hybrid.py --corpus sample/pcfg3_10k.txt --em-iters 30 --em-batch 500
"""

import argparse
import collections
import csv
import time
import numpy as np
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────────────
N_PRETERM = 26
PRETERM_NAMES = [chr(ord('A') + i) for i in range(N_PRETERM)]
TERM_TO_IDX = {chr(ord('a') + i): i for i in range(N_PRETERM)}


def sym_name(idx):
    if idx < N_PRETERM:
        return PRETERM_NAMES[idx]
    if idx == N_PRETERM:
        return "S"
    return f"NT{idx - N_PRETERM}"


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Corpus loading
# ═════════════════════════════════════════════════════════════════════════════

def load_corpus(path):
    sents = []
    with open(path) as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sents.append(np.array([TERM_TO_IDX[t] for t in toks], dtype=np.int32))
    return sents


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — BPE structure discovery
# ═════════════════════════════════════════════════════════════════════════════

def bpe_discover(sentences, max_merges=60, min_count=20):
    """
    Greedy pair-merging on preterminal sequences.
    Returns discovered rules and the final merged corpus.
    """
    corpus = [s.tolist() for s in sentences]
    next_id = N_PRETERM          # 0..25 are preterminals
    rules = []                   # (new_id, left_child, right_child)
    counts = []

    for step in range(max_merges):
        pair_freq = collections.Counter()
        for sent in corpus:
            for i in range(len(sent) - 1):
                pair_freq[(sent[i], sent[i + 1])] += 1
        if not pair_freq:
            break

        (left, right), freq = pair_freq.most_common(1)[0]
        if freq < min_count:
            break

        new_id = next_id
        next_id += 1

        new_corpus = []
        for sent in corpus:
            merged = []
            i = 0
            while i < len(sent):
                if i < len(sent) - 1 and sent[i] == left and sent[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(sent[i])
                    i += 1
            new_corpus.append(merged)
        corpus = new_corpus

        rules.append((new_id, left, right))
        counts.append(freq)
        print(f"  [{step+1:3d}] {sym_name(new_id):>6s} -> "
              f"{sym_name(left):>6s} {sym_name(right):<6s}  freq={freq}")

    n_nt = next_id - N_PRETERM
    return rules, counts, n_nt, corpus


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — PCFG with Inside-Outside EM
# ═════════════════════════════════════════════════════════════════════════════

class PCFG:
    """
    CNF grammar.

    Symbol indices:
        0 .. 25              preterminals  (A .. Z, fixed: A->a etc.)
        26                   S  (start symbol)
        27 .. 26 + n_nt - 1  other nonterminals

    Binary rules stored in a dense tensor:
        rules[a, b, c] = P( NT_{a+26}  ->  Sym_b  Sym_c )
        shape = (n_nt, N, N)   where N = 26 + n_nt
    """

    def __init__(self, n_nt):
        self.n_nt = n_nt
        self.N = N_PRETERM + n_nt
        self.S = N_PRETERM
        self.rules = np.zeros((n_nt, self.N, self.N), dtype=np.float64)

    # ── initialisation helpers ───────────────────────────────────────────

    def init_from_bpe(self, bpe_rules, bpe_counts, noise=1e-3):
        """
        Seed rule tensor from BPE evidence.

        BPE assigns IDs starting at 26 (== S).  The *first* BPE merge
        is the most frequent bottom-level pair; the *last* merge is the
        highest-level construct.  We remap so that the last BPE merge
        occupies index 26 (= S), and shift others up by one.
        """
        rng = np.random.default_rng(42)
        self.rules[:] = noise * rng.random(self.rules.shape)

        n_bpe = len(bpe_rules)
        if n_bpe == 0:
            self._normalise()
            return

        # Build remapping: last BPE symbol → S (26), others shift
        # BPE symbols are 26 .. 26+n_bpe-1
        last_bpe_id = bpe_rules[-1][0]   # highest-level merge
        remap = {}
        for i in range(N_PRETERM):
            remap[i] = i                  # preterminals unchanged
        # Map last BPE symbol to S
        remap[last_bpe_id] = self.S
        # Map the rest, filling 27, 28, ...
        fill = N_PRETERM + 1
        for new_id, _, _ in bpe_rules:
            if new_id == last_bpe_id:
                continue
            remap[new_id] = fill
            fill += 1

        for (new_id, left, right), cnt in zip(bpe_rules, bpe_counts):
            a_new = remap.get(new_id)
            l_new = remap.get(left)
            r_new = remap.get(right)
            if a_new is None or l_new is None or r_new is None:
                continue
            a_local = a_new - N_PRETERM
            if 0 <= a_local < self.n_nt and l_new < self.N and r_new < self.N:
                self.rules[a_local, l_new, r_new] += cnt

        self._normalise()

    def init_random(self, seed=42):
        rng = np.random.default_rng(seed)
        self.rules[:] = rng.random(self.rules.shape)
        self._normalise()

    def _normalise(self):
        for a in range(self.n_nt):
            s = self.rules[a].sum()
            if s > 0:
                self.rules[a] /= s

    # ── CYK inside algorithm ────────────────────────────────────────────

    def inside(self, sent):
        n = len(sent)
        alpha = np.zeros((self.N, n, n))
        for i in range(n):
            alpha[sent[i], i, i] = 1.0

        for width in range(2, n + 1):
            for i in range(n - width + 1):
                j = i + width - 1
                for k in range(i, j):
                    lv = alpha[:, i, k]          # (N,)
                    rv = alpha[:, k + 1, j]      # (N,)
                    bc = np.outer(lv, rv)         # (N, N)
                    # sum over b,c for each nonterminal a
                    alpha[N_PRETERM:, i, j] += np.einsum(
                        'abc,bc->a', self.rules, bc)
        return alpha

    # ── outside algorithm ────────────────────────────────────────────────

    def outside(self, sent, alpha):
        n = len(sent)
        beta = np.zeros((self.N, n, n))
        beta[self.S, 0, n - 1] = 1.0

        for width in range(n, 1, -1):
            for i in range(n - width + 1):
                j = i + width - 1
                pb = beta[N_PRETERM:, i, j]       # (n_nt,)
                if not np.any(pb):
                    continue
                for k in range(i, j):
                    r_a = alpha[:, k + 1, j]      # (N,)
                    l_a = alpha[:, i, k]           # (N,)

                    # propagate to left child at [i, k]
                    # beta[l, i, k] += Σ_a pb[a] · Σ_r rules[a,l,r]·r_a[r]
                    wr = np.einsum('alr,r->al', self.rules, r_a)
                    beta[:, i, k] += pb @ wr      # (N,)

                    # propagate to right child at [k+1, j]
                    wl = np.einsum('alr,l->ar', self.rules, l_a)
                    beta[:, k + 1, j] += pb @ wl  # (N,)
        return beta

    # ── one EM iteration ─────────────────────────────────────────────────

    def em_step(self, sentences, max_len=30):
        counts = np.zeros_like(self.rules)
        ll = 0.0
        used = 0

        for sent in sentences:
            n = len(sent)
            if n < 2 or n > max_len:
                continue

            alpha = self.inside(sent)
            sp = alpha[self.S, 0, n - 1]
            if sp < 1e-300:
                continue

            ll += np.log(sp)
            used += 1
            beta = self.outside(sent, alpha)

            for width in range(2, n + 1):
                for i in range
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import math
import os
import time
from pathlib import Path

import numpy as np


BASE_PRETERMS = [chr(ord("A") + i) for i in range(26)]
BASE_PRETERMS[ord("s") - ord("a")] = "PT_S"
WORD_TO_SYMBOL = {chr(ord("a") + i): BASE_PRETERMS[i] for i in range(26)}
SYMBOL_TO_WORD = {symbol: word for word, symbol in WORD_TO_SYMBOL.items()}

LEXICAL_PAIR_SYMBOLS = collections.OrderedDict(
    [
        (("A", "B"), "LP_AB"),
        (("A", "M"), "LP_AM"),
        (("C", "D"), "LP_CD"),
        (("E", "F"), "LP_EF"),
        (("E", "I"), "LP_EI"),
        (("G", "H"), "LP_GH"),
        (("I", "J"), "LP_IJ"),
        (("K", "L"), "LP_KL"),
        (("M", "N"), "LP_MN"),
        (("O", "P"), "LP_OP"),
        (("O", "U"), "LP_OU"),
        (("PT_S", "T"), "LP_ST"),
        (("Q", "R"), "LP_QR"),
        (("U", "V"), "LP_UV"),
        (("W", "X"), "LP_WX"),
        (("Y", "Z"), "LP_YZ"),
    ]
)
PAIR_SYMBOL_TO_RHS = {symbol: pair for pair, symbol in LEXICAL_PAIR_SYMBOLS.items()}
FAMILY_GROUPS = collections.OrderedDict(
    [
        ("F_ABCD", ["LP_AB", "LP_CD"]),
        ("F_EFGH", ["LP_EF", "LP_GH"]),
        ("F_EIOU", ["LP_EI", "LP_OU"]),
        ("F_IJKL", ["LP_IJ", "LP_KL"]),
        ("F_MNOP", ["LP_MN", "LP_OP"]),
        ("F_QRST", ["LP_QR", "LP_ST"]),
        ("F_UVWX", ["LP_UV", "LP_WX"]),
        ("F_AM", ["LP_AM"]),
        ("F_YZ", ["LP_YZ"]),
    ]
)
DEFAULT_PHASES = [
    {"name": "short_warmup", "max_len": 10, "iters": 8, "batch": 2000, "prune_every": 2, "soft_prune": 0.0005},
    {"name": "medium_expand", "max_len": 16, "iters": 8, "batch": 2500, "prune_every": 4, "soft_prune": 0.0010},
    {"name": "long_refine", "max_len": 24, "iters": 10, "batch": 3000, "prune_every": 5, "soft_prune": 0.0015},
]


def build_pair_to_family() -> dict[str, str]:
    pair_to_family: dict[str, str] = {}
    for family_symbol, pair_symbols in FAMILY_GROUPS.items():
        for pair_symbol in pair_symbols:
            if pair_symbol in pair_to_family:
                raise ValueError(f"duplicate family assignment for {pair_symbol}")
            pair_to_family[pair_symbol] = family_symbol

    missing_pairs = [symbol for symbol in PAIR_SYMBOL_TO_RHS if symbol not in pair_to_family]
    if missing_pairs:
        raise ValueError(f"unassigned lexical pairs: {missing_pairs}")
    return pair_to_family


PAIR_TO_FAMILY = build_pair_to_family()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_seed_values(seed_values_text: str | None, fallback_seed: int) -> list[int]:
    if seed_values_text:
        values = parse_int_list(seed_values_text)
        if values:
            return values
    return [fallback_seed]


def load_symbol_corpus(path: Path) -> list[list[str]]:
    corpus: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                corpus.append([WORD_TO_SYMBOL[tok] for tok in toks])
    return corpus


def segment_to_pair_symbols(sentence: list[str]) -> tuple[list[str] | None, int | None]:
    segmented: list[str] = []
    i = 0
    while i < len(sentence):
        pair = (sentence[i], sentence[i + 1]) if i + 1 < len(sentence) else None
        if pair in LEXICAL_PAIR_SYMBOLS:
            segmented.append(LEXICAL_PAIR_SYMBOLS[pair])
            i += 2
        else:
            return None, i
    return segmented, None


def prepare_family_grouped_corpus(corpus_path: Path) -> dict:
    raw_corpus = load_symbol_corpus(corpus_path)
    pair_corpus: list[list[str]] = []
    family_corpus: list[list[str]] = []
    failed_segmentations = 0
    pair_symbol_counts: collections.Counter[str] = collections.Counter()
    family_symbol_counts: collections.Counter[str] = collections.Counter()
    family_pair_choice_counts = {
        family_symbol: collections.Counter()
        for family_symbol in FAMILY_GROUPS
    }

    for sent in raw_corpus:
        segmented, _ = segment_to_pair_symbols(sent)
        if segmented is None:
            failed_segmentations += 1
            continue
        pair_corpus.append(segmented)
        pair_symbol_counts.update(segmented)
        family_sent = [PAIR_TO_FAMILY[pair_symbol] for pair_symbol in segmented]
        family_corpus.append(family_sent)
        family_symbol_counts.update(family_sent)
        for pair_symbol, family_symbol in zip(segmented, family_sent):
            family_pair_choice_counts[family_symbol][pair_symbol] += 1

    observed_symbols = [family_symbol for family_symbol in FAMILY_GROUPS if family_symbol_counts[family_symbol] > 0]
    family_to_pair_probs = collections.OrderedDict()
    for family_symbol, pair_symbols in FAMILY_GROUPS.items():
        counts = family_pair_choice_counts[family_symbol]
        total = sum(counts.values())
        if total == 0:
            continue
        family_to_pair_probs[family_symbol] = collections.OrderedDict(
            (pair_symbol, counts[pair_symbol] / total)
            for pair_symbol in pair_symbols
            if counts[pair_symbol] > 0
        )

    return {
        "raw_count": len(raw_corpus),
        "covered_count": len(pair_corpus),
        "dropped_count": failed_segmentations,
        "coverage_pct": 100.0 * len(pair_corpus) / max(len(raw_corpus), 1),
        "pair_corpus": pair_corpus,
        "family_corpus": family_corpus,
        "family_to_pair_probs": family_to_pair_probs,
        "observed_symbols": observed_symbols,
    }


class GeneralPCFG_EM:
    def __init__(self, observed_symbols: list[str], n_nt: int):
        self.observed_symbols = list(observed_symbols)
        self.obs_to_idx = {sym: i for i, sym in enumerate(self.observed_symbols)}
        self.n_obs = len(self.observed_symbols)
        self.n_nt = n_nt
        self.N = self.n_obs + n_nt
        self.S = self.n_obs
        self.rules = np.zeros((n_nt, self.N, self.N), dtype=np.float64)

    def sym_name(self, idx: int) -> str:
        if idx < self.n_obs:
            return self.observed_symbols[idx]
        if idx == self.S:
            return "S"
        return f"NT{idx - self.n_obs}"

    def encode_corpus(self, symbol_corpus: list[list[str]]) -> list[np.ndarray]:
        encoded: list[np.ndarray] = []
        for sent in symbol_corpus:
            encoded.append(np.array([self.obs_to_idx[sym] for sym in sent], dtype=np.int32))
        return encoded

    def init_random(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.rules[:] = rng.random(self.rules.shape)
        self._normalise()

    def clone(self) -> "GeneralPCFG_EM":
        copied = GeneralPCFG_EM(self.observed_symbols, self.n_nt)
        copied.rules = self.rules.copy()
        return copied

    def _normalise(self) -> None:
        for a in range(self.n_nt):
            total = self.rules[a].sum()
            if total > 0:
                self.rules[a] /= total

    def inside(self, sent: np.ndarray) -> np.ndarray:
        n = len(sent)
        alpha = np.zeros((self.N, n, n), dtype=np.float64)
        for i in range(n):
            alpha[sent[i], i, i] = 1.0

        for width in range(2, n + 1):
            for i in range(n - width + 1):
                j = i + width - 1
                for k in range(i, j):
                    left = alpha[:, i, k]
                    right = alpha[:, k + 1, j]
                    outer = np.outer(left, right)
                    alpha[self.n_obs :, i, j] += np.einsum("abc,bc->a", self.rules, outer)
        return alpha

    def outside(self, sent: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        n = len(sent)
        beta = np.zeros((self.N, n, n), dtype=np.float64)
        beta[self.S, 0, n - 1] = 1.0

        for width in range(n, 1, -1):
            for i in range(n - width + 1):
                j = i + width - 1
                parent_beta = beta[self.n_obs :, i, j]
                if not np.any(parent_beta):
                    continue
                for k in range(i, j):
                    left_alpha = alpha[:, i, k]
                    right_alpha = alpha[:, k + 1, j]
                    wr = np.einsum("alr,r->al", self.rules, right_alpha)
                    beta[:, i, k] += parent_beta @ wr
                    wl = np.einsum("alr,l->ar", self.rules, left_alpha)
                    beta[:, k + 1, j] += parent_beta @ wl
        return beta

    def em_step(self, sentences: list[np.ndarray], max_len: int = 30) -> tuple[float, int]:
        counts = np.zeros_like(self.rules)
        ll = 0.0
        used = 0

        for sent in sentences:
            n = len(sent)
            if n < 2 or n > max_len:
                continue

            alpha = self.inside(sent)
            sentence_prob = alpha[self.S, 0, n - 1]
            if sentence_prob < 1e-300:
                continue

            ll += np.log(sentence_prob)
            used += 1
            beta = self.outside(sent, alpha)

            for width in range(2, n + 1):
                for i in range(n - width + 1):
                    j = i + width - 1
                    parent_beta = beta[self.n_obs :, i, j]
                    if not np.any(parent_beta):
                        continue
                    for k in range(i, j):
                        left = alpha[:, i, k]
                        right = alpha[:, k + 1, j]
                        outer = np.outer(left, right)
                        counts += (parent_beta[:, None, None] * self.rules * outer[None, :, :]) / sentence_prob

        for a in range(self.n_nt):
            total = counts[a].sum()
            if total > 0:
                self.rules[a] = counts[a] / total

        return ll, used

    def prune(self, threshold: float = 0.02) -> None:
        self.rules[self.rules < threshold] = 0.0
        self._normalise()

    def count_rules(self, threshold: float = 1e-10) -> int:
        return int(np.sum(self.rules > threshold))


def run_schedule(
    model: GeneralPCFG_EM,
    encoded_corpus: list[np.ndarray],
    phases: list[dict],
    seed: int = 42,
    verbose: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    sorted_corpus = sorted(encoded_corpus, key=len)
    history: list[dict] = []
    global_iter = 0

    for phase in phases:
        max_len = phase["max_len"]
        iters = phase["iters"]
        batch = phase.get("batch", 0)
        prune_every = phase.get("prune_every", 0)
        soft_prune = phase.get("soft_prune", 0.0)
        eligible = [sent for sent in sorted_corpus if 2 <= len(sent) <= max_len]

        if verbose:
            print(
                f"[n_nt={model.n_nt}] phase={phase['name']} max_len={max_len} "
                f"iters={iters} batch={batch or 'all'} eligible={len(eligible)}"
            )

        for phase_iter in range(1, iters + 1):
            global_iter += 1
            if batch and batch < len(eligible):
                idxs = rng.choice(len(eligible), size=batch, replace=False)
                batch_sents = [eligible[i] for i in idxs]
            else:
                batch_sents = eligible

            t0 = time.time()
            ll, used = model.em_step(batch_sents, max_len=max_len)
            dt = time.time() - t0
            avg_ll = ll / max(used, 1)
            nr = model.count_rules()
            history.append(
                {
                    "iter": global_iter,
                    "phase": phase["name"],
                    "phase_iter": phase_iter,
                    "avg_ll": avg_ll,
                    "rules": nr,
                    "used": used,
                    "time": dt,
                }
            )

            if verbose:
                print(
                    f"  iter={global_iter:02d} phase_iter={phase_iter:02d}/{iters} "
                    f"avg_LL={avg_ll:8.3f} used={used:4d} rules={nr:5d} {dt:5.1f}s"
                )

            if prune_every and soft_prune > 0 and phase_iter % prune_every == 0:
                before = model.count_rules()
                model.prune(threshold=soft_prune)
                after = model.count_rules()
                if verbose:
                    print(f"    early prune: {before} -> {after}")

    return history


def evaluate_inside(model: GeneralPCFG_EM, encoded_corpus: list[np.ndarray]) -> dict:
    parsed = 0
    failed = 0
    total_log2_prob = 0.0

    for sent in encoded_corpus:
        alpha = model.inside(sent)
        prob = alpha[model.S, 0, len(sent) - 1]
        if prob > 1e-300:
            parsed += 1
            total_log2_prob += math.log2(prob)
        else:
            failed += 1

    total = parsed + failed
    return {
        "parsed": parsed,
        "failed": failed,
        "total": total,
        "coverage": 100.0 * parsed / max(total, 1),
        "avg_log2_prob": total_log2_prob / max(parsed, 1),
    }


def evaluate_exact(
    model: GeneralPCFG_EM,
    family_corpus: list[list[str]],
    pair_corpus: list[list[str]],
    family_to_pair_probs: dict[str, collections.OrderedDict[str, float]],
) -> dict:
    encoded_eval = model.encode_corpus(family_corpus)
    parsed = 0
    failed = 0
    total_log2_prob = 0.0

    for family_sent, pair_sent, encoded_sent in zip(family_corpus, pair_corpus, encoded_eval):
        alpha = model.inside(encoded_sent)
        family_prob = alpha[model.S, 0, len(encoded_sent) - 1]
        if family_prob <= 1e-300:
            failed += 1
            continue

        lexical_log2 = 0.0
        lexical_ok = True
        for family_symbol, pair_symbol in zip(family_sent, pair_sent):
            pair_prob = family_to_pair_probs[family_symbol].get(pair_symbol, 0.0)
            if pair_prob <= 1e-300:
                lexical_ok = False
                break
            lexical_log2 += math.log2(pair_prob)

        if lexical_ok:
            parsed += 1
            total_log2_prob += math.log2(family_prob) + lexical_log2
        else:
            failed += 1

    total = parsed + failed
    return {
        "parsed": parsed,
        "failed": failed,
        "total": total,
        "coverage": 100.0 * parsed / max(total, 1),
        "avg_log2_prob": total_log2_prob / max(parsed, 1),
    }


def task3_rule_credit(binary_rules: int, true_binary_rules: int = 50) -> float:
    # README: full rule credit up to 150% of ground truth, then linearly down to zero at 200%.
    full_credit_limit = int(true_binary_rules * 1.5)
    zero_credit_limit = int(true_binary_rules * 2.0)
    if binary_rules <= full_credit_limit:
        return 1.0
    if binary_rules >= zero_credit_limit:
        return 0.0
    return (zero_credit_limit - binary_rules) / max(zero_credit_limit - full_credit_limit, 1)


def export_expanded_csv(
    model: GeneralPCFG_EM,
    output_path: Path,
    family_to_pair_probs: dict[str, collections.OrderedDict[str, float]],
    threshold: float = 1e-10,
) -> None:
    rows = []
    rule_id = 1

    for base_symbol in BASE_PRETERMS:
        rows.append(
            {
                "ID": rule_id,
                "LHS": base_symbol,
                "LHS Type": "preterminal",
                "RHS": SYMBOL_TO_WORD[base_symbol],
                "Probability": 1.0,
            }
        )
        rule_id += 1

    for family_symbol, rhs_probs in family_to_pair_probs.items():
        if family_symbol not in model.obs_to_idx:
            continue
        for pair_symbol, prob in rhs_probs.items():
            left, right = PAIR_SYMBOL_TO_RHS[pair_symbol]
            rows.append(
                {
                    "ID": rule_id,
                    "LHS": family_symbol,
                    "LHS Type": "nonterminal",
                    "RHS": f"{left} {right}",
                    "Probability": round(float(prob), 6),
                }
            )
            rule_id += 1

    for a in range(model.n_nt):
        for b in range(model.N):
            for c in range(model.N):
                prob = model.rules[a, b, c]
                if prob > threshold:
                    rows.append(
                        {
                            "ID": rule_id,
                            "LHS": model.sym_name(a + model.n_obs),
                            "LHS Type": "nonterminal",
                            "RHS": f"{model.sym_name(b)} {model.sym_name(c)}",
                            "Probability": round(float(prob), 6),
                        }
                    )
                    rule_id += 1

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "LHS", "LHS Type", "RHS", "Probability"])
        writer.writeheader()
        writer.writerows(rows)


def run_single_experiment(config: dict) -> dict:
    n_nt = config["n_nt"]
    seed = config["seed"]
    corpus_path = Path(config["corpus"])
    phases = config["phases"]
    max_train_sentences = config["max_train_sentences"]
    eval_sentences = config["eval_sentences"]
    final_prune = config["final_prune"]
    eval_mode = config["eval_mode"]
    export_dir = config["export_dir"]
    verbose = config["verbose"]

    started = time.time()

    try:
        prepared = prepare_family_grouped_corpus(corpus_path)
        pair_corpus = prepared["pair_corpus"]
        family_corpus = prepared["family_corpus"]
        family_to_pair_probs = prepared["family_to_pair_probs"]
        observed_symbols = prepared["observed_symbols"]

        if max_train_sentences > 0:
            train_corpus = family_corpus[:max_train_sentences]
        else:
            train_corpus = family_corpus

        if eval_sentences > 0:
            eval_family_corpus = family_corpus[:eval_sentences]
            eval_pair_corpus = pair_corpus[:eval_sentences]
        else:
            eval_family_corpus = family_corpus
            eval_pair_corpus = pair_corpus

        model = GeneralPCFG_EM(observed_symbols, n_nt)
        model.init_random(seed=seed)
        encoded_train = model.encode_corpus(train_corpus)
        run_schedule(model, encoded_train, phases, seed=seed, verbose=verbose)

        rules_before_prune = model.count_rules()
        model.prune(threshold=final_prune)
        rules_after_prune = model.count_rules()
        rule_credit = task3_rule_credit(rules_after_prune)

        if eval_mode == "inside":
            encoded_eval = model.encode_corpus(eval_family_corpus)
            metrics = evaluate_inside(model, encoded_eval)
        else:
            metrics = evaluate_exact(model, eval_family_corpus, eval_pair_corpus, family_to_pair_probs)

        if export_dir:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            csv_path = export_path / f"pcfg3_lpsecond_seed{seed}_nnt{n_nt}.csv"
            export_expanded_csv(model, csv_path, family_to_pair_probs)
        else:
            csv_path = None

        elapsed = time.time() - started
        return {
            "n_nt": n_nt,
            "seed": seed,
            "status": "PASS",
            "coverage": metrics["coverage"],
            "parsed": metrics["parsed"],
            "failed": metrics["failed"],
            "total": metrics["total"],
            "avg_log2_prob": metrics["avg_log2_prob"],
            "rules_before_prune": rules_before_prune,
            "rules_after_prune": rules_after_prune,
            "rule_credit": rule_credit,
            "observed_symbols": len(observed_symbols),
            "train_sentences": len(train_corpus),
            "eval_sentences": len(eval_family_corpus),
            "segmented_sentences": prepared["covered_count"],
            "dropped_sentences": prepared["dropped_count"],
            "segmentation_coverage": prepared["coverage_pct"],
            "elapsed_sec": elapsed,
            "export_csv": str(csv_path) if csv_path else "",
            "error": "",
        }
    except Exception as exc:
        elapsed = time.time() - started
        return {
            "n_nt": n_nt,
            "seed": seed,
            "status": "ERROR",
            "coverage": 0.0,
            "parsed": 0,
            "failed": 0,
            "total": 0,
            "avg_log2_prob": float("-inf"),
            "rules_before_prune": 0,
            "rules_after_prune": 0,
            "rule_credit": 0.0,
            "observed_symbols": 0,
            "train_sentences": 0,
            "eval_sentences": 0,
            "segmented_sentences": 0,
            "dropped_sentences": 0,
            "segmentation_coverage": 0.0,
            "elapsed_sec": elapsed,
            "export_csv": "",
            "error": repr(exc),
        }


def print_summary(results: list[dict], target_coverage: float) -> None:
    print("")
    print(
        "seed  N_NT  coverage  parsed/total  avg_log2  rules  rule%  sec    result"
    )
    print(
        "----  ----  --------  ------------  --------  -----  -----  -----  ------"
    )
    for row in results:
        if row["status"] != "PASS":
            result = f"ERROR {row['error']}"
            print(
                f"{row['seed']:>4d}  {row['n_nt']:>4d}  {'0.00%':>8s}  {'0/0':>12s}  {'-inf':>8s}  "
                f"{0:>5d}  {'0.0':>5s}  {row['elapsed_sec']:>5.1f}  {result}"
            )
            continue

        result = "PASS" if row["coverage"] >= target_coverage else "FAIL"
        print(
            f"{row['seed']:>4d}  {row['n_nt']:>4d}  {row['coverage']:7.2f}%  "
            f"{row['parsed']:>5d}/{row['total']:<6d}  "
            f"{row['avg_log2_prob']:8.4f}  {row['rules_after_prune']:>5d}  "
            f"{100.0 * row['rule_credit']:5.1f}  {row['elapsed_sec']:>5.1f}  {result}"
        )


def write_summary_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = [
        "seed",
        "n_nt",
        "status",
        "coverage",
        "parsed",
        "failed",
        "total",
        "avg_log2_prob",
        "rules_before_prune",
        "rules_after_prune",
        "rule_credit",
        "observed_symbols",
        "train_sentences",
        "eval_sentences",
        "segmented_sentences",
        "dropped_sentences",
        "segmentation_coverage",
        "elapsed_sec",
        "export_csv",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_parallel_configs(configs: list[dict], jobs: int, executor_mode: str) -> list[dict]:
    if jobs == 1:
        return [run_single_experiment(cfg) for cfg in configs]

    def collect_with_executor(executor_cls):
        results = []
        with executor_cls(max_workers=jobs) as executor:
            futures = {
                executor.submit(run_single_experiment, cfg): (cfg["seed"], cfg["n_nt"])
                for cfg in configs
            }
            for future in cf.as_completed(futures):
                result = future.result()
                results.append(result)
                if result["status"] == "PASS":
                    print(
                        f"finished seed={result['seed']} n_nt={result['n_nt']}: "
                        f"coverage={result['coverage']:.2f}% parsed={result['parsed']}/{result['total']} "
                        f"rules={result['rules_after_prune']} sec={result['elapsed_sec']:.1f}"
                    )
                else:
                    print(
                        f"finished seed={result['seed']} n_nt={result['n_nt']}: "
                        f"ERROR {result['error']}"
                    )
        return results

    if executor_mode == "thread":
        print("Executor: thread")
        return collect_with_executor(cf.ThreadPoolExecutor)

    if executor_mode == "process":
        print("Executor: process")
        return collect_with_executor(cf.ProcessPoolExecutor)

    try:
        print("Executor: process (auto)")
        return collect_with_executor(cf.ProcessPoolExecutor)
    except PermissionError as exc:
        print(f"Process executor unavailable ({exc}). Falling back to thread executor.")
        return collect_with_executor(cf.ThreadPoolExecutor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel N_NT sweep for the Task 3 lexical-pair-second family-grouped grammar."
    )
    parser.add_argument(
        "--corpus",
        default="sample/pcfg3_10k.txt",
        help="Original Task 3 corpus.",
    )
    parser.add_argument(
        "--n-nt-values",
        default="8,10,12,14,16",
        help="Comma-separated N_NT values to sweep.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fallback random seed used when --seed-values is not provided.",
    )
    parser.add_argument(
        "--seed-values",
        default="",
        help="Comma-separated random seeds to sweep. If provided, overrides --seed.",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        default=0,
        help="Limit training to the first N family-grouped sentences. 0 uses all.",
    )
    parser.add_argument(
        "--eval-sentences",
        type=int,
        default=0,
        help="Limit evaluation to the first N pair-resolved sentences. 0 uses all.",
    )
    parser.add_argument(
        "--final-prune",
        type=float,
        default=0.02,
        help="Final prune threshold applied before evaluation.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["exact", "inside", "parser"],
        default="exact",
        help="Use exact pair-resolved evaluation, or faster family-only inside evaluation.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=95.0,
        help="Coverage target used to mark PASS/FAIL in the summary.",
    )
    parser.add_argument(
        "--executor",
        choices=["auto", "process", "thread"],
        default="auto",
        help="Parallel executor backend.",
    )
    parser.add_argument(
        "--summary-csv",
        default="task3_lexical_pair_sweep_results.csv",
        help="Where to write the ranked summary CSV.",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="Optional directory for per-config exported grammars.",
    )
    parser.add_argument(
        "--verbose-workers",
        action="store_true",
        help="Print detailed phase logs inside each worker.",
    )
    args = parser.parse_args()

    n_nt_values = parse_int_list(args.n_nt_values)
    seed_values = parse_seed_values(args.seed_values, args.seed)
    if not n_nt_values:
        raise SystemExit("No N_NT values provided.")
    if not seed_values:
        raise SystemExit("No seed values provided.")

    configs = [
        {
            "n_nt": n_nt,
            "seed": seed,
            "corpus": args.corpus,
            "phases": DEFAULT_PHASES,
            "max_train_sentences": args.max_train_sentences,
            "eval_sentences": args.eval_sentences,
            "final_prune": args.final_prune,
            "eval_mode": args.eval_mode,
            "export_dir": args.export_dir,
            "verbose": args.verbose_workers,
        }
        for seed in seed_values
        for n_nt in n_nt_values
    ]

    print("Family-grouped segmentation coverage is fixed across runs.")
    print(f"Running N_NT sweep: {n_nt_values}")
    print(f"Running seed sweep: {seed_values}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Jobs: {args.jobs}")
    print(f"Executor request: {args.executor}")
    if args.max_train_sentences:
        print(f"Train subset: first {args.max_train_sentences} family-grouped sentences")
    else:
        print("Train subset: full family-grouped corpus")
    if args.eval_sentences:
        print(f"Eval subset: first {args.eval_sentences} pair-resolved sentences")
    else:
        print("Eval subset: full pair-resolved corpus")

    results = run_parallel_configs(configs, args.jobs, args.executor)

    results.sort(
        key=lambda row: (
            row["status"] == "PASS",
            row.get("coverage", 0.0),
            row.get("avg_log2_prob", float("-inf")),
            row.get("rule_credit", 0.0),
            -row.get("rules_after_prune", 0),
        ),
        reverse=True,
    )
    print_summary(results, args.target_coverage)

    summary_csv = args.summary_csv
    if summary_csv == "task3_lexical_pair_sweep_results.csv":
        if len(seed_values) == 1:
            summary_csv = f"task3_lexical_pair_sweep_results_seed{seed_values[0]}.csv"
        else:
            summary_csv = "task3_lexical_pair_sweep_results_multiseed.csv"

    summary_path = Path(summary_csv)
    write_summary_csv(results, summary_path)
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

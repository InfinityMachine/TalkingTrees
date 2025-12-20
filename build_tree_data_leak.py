import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import smolagents
from huggingface_hub import login
from sklearn.model_selection import train_test_split

import dataset_descriptions
import prompting
import proxy_api_model
import tree_agent
from task import get_task_variables, metric_func_by_task

MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 30


# -----------------------
# Helpers
# -----------------------
def _stable_seed(*parts: Any, mod: int = 2**32 - 1) -> int:
    h = hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _normalize_multiclass_proba(y_pred: np.ndarray) -> np.ndarray:
    denom = y_pred.sum(axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return y_pred / denom


def _metric_higher_is_better(metric_name: str) -> bool:
    # Heuristic; override if you have a better mapping.
    m = metric_name.lower()
    if any(k in m for k in ["loss", "rmse", "mae", "mse", "error"]):
        return False
    return True


# -----------------------
# Corruption
# -----------------------
@dataclass
class CorruptionConfig:
    # mild
    mild_mask_rate: float = 0.05  # fraction of entries masked per selected column
    mild_noise_std: float = 0.05  # noise in standardized units (numeric cols)
    mild_scale_low: float = 0.7  # per-column scaling factors
    mild_scale_high: float = 1.3
    mild_rotate_frac: float = (
        0.30  # rotate only a subset of numeric columns (keeps it "mild")
    )

    # some
    some_shuffle_col_frac: float = 0.50  # fraction of columns to shuffle within

    # complete
    complete_shuffle_all: bool = True  # shuffle all columns + permute labels


class TrainValCorruptor:
    """
    Fits on X_train only; then transforms X_train, X_val, and X_test (for scope B).
    Noise levels:
      0: none
      1: mild (subset rotation on numeric cols + scaling + noise + masking)
      2: some (shuffle values within subset of columns)
      3: complete (shuffle all columns + permute labels in train/val)
    """

    def __init__(self, noise_level: int, seed: int, cfg: CorruptionConfig):
        self.noise_level = int(noise_level)
        self.seed = int(seed)
        self.cfg = cfg

        self.cols_all_: List[str] = []
        self.cols_num_: List[str] = []
        self.cols_rot_: List[str] = []
        self.cols_shuffle_: List[str] = []

        # mild params
        self.scale_: Dict[str, float] = {}
        self.rot_Q_: Optional[np.ndarray] = None
        self.rot_mean_: Optional[np.ndarray] = None
        self.rot_std_: Optional[np.ndarray] = None

    def fit(self, X_train: pd.DataFrame) -> "TrainValCorruptor":
        rng = np.random.default_rng(self.seed)
        self.cols_all_ = list(X_train.columns)
        self.cols_num_ = [c for c in self.cols_all_ if _is_numeric(X_train[c])]

        if self.noise_level == 1:
            # pick numeric columns to rotate (subset to keep it mild)
            if len(self.cols_num_) >= 2 and self.cfg.mild_rotate_frac > 0:
                k = max(2, int(round(self.cfg.mild_rotate_frac * len(self.cols_num_))))
                self.cols_rot_ = list(
                    rng.choice(
                        self.cols_num_, size=min(k, len(self.cols_num_)), replace=False
                    )
                )
                d = len(self.cols_rot_)
                A = rng.normal(size=(d, d))
                Q, R = np.linalg.qr(A)
                # stabilize sign a bit
                diag = np.sign(np.diag(R))
                diag[diag == 0] = 1.0
                Q = Q * diag
                self.rot_Q_ = Q

                Xr = X_train[self.cols_rot_].astype(float).to_numpy()
                self.rot_mean_ = np.nanmean(Xr, axis=0)
                self.rot_std_ = np.nanstd(Xr, axis=0)
                self.rot_std_ = np.where(self.rot_std_ == 0, 1.0, self.rot_std_)

            # per-column scaling for numeric cols
            for c in self.cols_num_:
                self.scale_[c] = float(
                    rng.uniform(self.cfg.mild_scale_low, self.cfg.mild_scale_high)
                )

        elif self.noise_level == 2:
            # shuffle subset of columns
            k = max(1, int(round(self.cfg.some_shuffle_col_frac * len(self.cols_all_))))
            self.cols_shuffle_ = list(
                rng.choice(
                    self.cols_all_, size=min(k, len(self.cols_all_)), replace=False
                )
            )

        elif self.noise_level == 3:
            self.cols_shuffle_ = self.cols_all_[:]  # shuffle all columns

        return self

    def transform_X(self, X: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
        if self.noise_level == 0:
            return X

        rng = np.random.default_rng(
            _stable_seed(self.seed, "X", self.noise_level, split_name, len(X))
        )
        Xc = X.copy()

        if self.noise_level == 1:
            # (a) rotate subset of numeric columns
            if self.rot_Q_ is not None and self.cols_rot_:
                Xr = Xc[self.cols_rot_].astype(float).to_numpy()
                Xr = np.nan_to_num(Xr, nan=0.0)
                Xs = (Xr - self.rot_mean_) / self.rot_std_
                Xrot = Xs @ self.rot_Q_
                # back to roughly original scale
                Xrot = Xrot * self.rot_std_ + self.rot_mean_
                for col in self.cols_rot_:
                    Xc[col] = Xc[col].astype(float)
                Xc.loc[:, self.cols_rot_] = Xrot

            # (b) scale + add noise on numeric columns
            for c in self.cols_num_:
                s = pd.to_numeric(Xc[c], errors="coerce").astype(float).to_numpy()
                s = s * self.scale_.get(c, 1.0)
                # noise in standardized-ish units
                col_std = np.nanstd(s)
                if not np.isfinite(col_std) or col_std == 0:
                    col_std = 1.0
                s = s + rng.normal(scale=self.cfg.mild_noise_std * col_std, size=len(s))
                Xc[c] = s

            # (c) mask values (all columns)
            if self.cfg.mild_mask_rate > 0:
                for c in self.cols_all_:
                    m = rng.random(len(Xc)) < self.cfg.mild_mask_rate
                    Xc.loc[m, c] = np.nan

            return Xc

        # noise_level 2 or 3: shuffle within columns
        for c in self.cols_shuffle_:
            vals = Xc[c].to_numpy()
            perm = rng.permutation(len(vals))
            Xc[c] = vals[perm]

        return Xc

    def transform_y_train_val(
        self, y_train: np.ndarray, y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.noise_level < 3:
            return y_train, y_val
        rng = np.random.default_rng(
            _stable_seed(self.seed, "y", self.noise_level, len(y_train), len(y_val))
        )
        return rng.permutation(y_train), rng.permutation(y_val)


# -----------------------
# Prompting (metadata modes)
# -----------------------
def make_prompt(
    *,
    dataset_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    metadata_mode: str,  # "true" | "none" | "wrong"
    dataset_descriptions,
    prompting,
    wrong_dataset_name: Optional[str] = None,
) -> str:
    if metadata_mode == "true":
        shown_name = dataset_name
        desc = dataset_descriptions.desc.get(dataset_name, "")
        desc = desc.format(
            num_samples=len(X_train), metric=prompting.metrics_by_task[task_type]
        )

    elif metadata_mode == "none":
        shown_name = "Dataset A"
        desc = "(No dataset description provided.)"

    elif metadata_mode == "wrong":
        shown_name = wrong_dataset_name or "Dataset B"
        wrong_desc = dataset_descriptions.desc.get(shown_name, "")
        if wrong_desc:
            desc = wrong_desc.format(
                num_samples=len(X_train), metric=prompting.metrics_by_task[task_type]
            )
        else:
            desc = "(Description intentionally misleading / unavailable.)"

    else:
        raise ValueError(f"Unknown metadata_mode={metadata_mode}")

    return f"""
Build the optimal decision tree for the '{shown_name}' dataset.
You are given access to 4 data variables in your python environment:
 - X_train, X_val are pandas dataframes with named feature columns (see below) that may need preprocessing;
 - y_train, y_val are numpy arrays (1d) with targets, also described below;

Dataset description (use it to form hypotheses):
{desc}

Here's one way you could construct before you begin editing it manually:
{prompting.starter_snippets_by_task[task_type]}

Now begin: view the data variables, preprocess as necessary, train a baseline tree, then propose the first hypothesis and start improving.
Focus on drawing conclusions from data, looking at the tree (e.g. via print) and using your own intuition about the problem for manual tree edits.
Quality is more important than speed: take as many steps as you need to get the best tree.
""".strip()


# -----------------------
# Main experiment
# -----------------------
def _load_completed_keys(results_jsonl_path: str) -> set:
    completed = set()
    try:
        with open(results_jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = (row["repeat"], row["noise_level"], row["metadata_mode"])
                    completed.add(key)
    except FileNotFoundError:
        pass
    return completed


from catboost import CatBoostClassifier, CatBoostRegressor


def _prepare_for_catboost(df, cat_features_idx, num_features_idx, num_fill_values=None):
    """
    Prepare DataFrame for CatBoost: handle NaN in categorical and numerical columns.
    Returns: (prepared_df, num_fill_values dict)
    """
    df = df.copy()

    for idx in cat_features_idx:
        col = df.columns[idx]
        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": "missing", "None": "missing", "": "missing"})
        )

    if num_fill_values is None:
        num_fill_values = {}
        for idx in num_features_idx:
            col = df.columns[idx]
            num_fill_values[col] = df[col].median()

    for idx in num_features_idx:
        col = df.columns[idx]
        df[col] = df[col].fillna(num_fill_values[col])

    return df, num_fill_values


def _run_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test_clean,
    X_test_corrupt,
    y_test,
    task_type,
    metric_func,
):
    """Run CatBoost and return (score_A, score_B) for clean and corrupt test sets."""
    cat_features_idx = [
        i
        for i, col in enumerate(X_train.columns)
        if X_train[col].dtype == "object" or X_train[col].dtype.name == "category"
    ]
    num_features_idx = [
        i
        for i, col in enumerate(X_train.columns)
        if X_train[col].dtype in ["int64", "float64", "int32", "float32"]
    ]

    X_train_cb, fill_vals = _prepare_for_catboost(
        X_train, cat_features_idx, num_features_idx
    )
    X_val_cb, _ = _prepare_for_catboost(
        X_val, cat_features_idx, num_features_idx, fill_vals
    )
    X_test_clean_cb, _ = _prepare_for_catboost(
        X_test_clean, cat_features_idx, num_features_idx, fill_vals
    )
    X_test_corrupt_cb, _ = _prepare_for_catboost(
        X_test_corrupt, cat_features_idx, num_features_idx, fill_vals
    )

    cat_features = cat_features_idx if cat_features_idx else None
    common_params = dict(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=0,
        cat_features=cat_features,
    )

    if task_type == "regression":
        model = CatBoostRegressor(**common_params)
    else:
        model = CatBoostClassifier(**common_params)

    model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), early_stopping_rounds=50)

    if task_type == "regression":
        preds_clean, preds_corrupt = (
            model.predict(X_test_clean_cb),
            model.predict(X_test_corrupt_cb),
        )
    else:
        preds_clean, preds_corrupt = (
            model.predict_proba(X_test_clean_cb),
            model.predict_proba(X_test_corrupt_cb),
        )

    return float(metric_func(y_test, preds_clean)), float(
        metric_func(y_test, preds_corrupt)
    )


def _run_baseline(y_train, y_test, task_type, metric_func, variant="soft"):
    """
    Naive baseline:
    - regression: mean prediction (variant ignored)
    - classification soft: class-frequency probabilities
    - classification hard: one-hot for majority class
    """
    if task_type == "regression":
        preds = np.full(len(y_test), np.mean(y_train))
    else:
        values, counts = np.unique(y_train, return_counts=True)
        n_classes = len(values)

        if variant == "soft":
            probs = counts / counts.sum()
        else:  # hard
            probs = np.zeros(n_classes)
            probs[np.argmax(counts)] = 1.0

        preds = np.tile(probs, (len(y_test), 1))

    score = float(metric_func(y_test, preds))
    return score, score  # identical for clean/corrupt (ignores features)


def run_leakage_sweep(
    *,
    dataset_name: str,
    model,
    n_repeats: int,
    noise_levels: List[int],
    metadata_modes: List[str],
    corruption_cfg: CorruptionConfig,
    results_jsonl_path: str,
    openml,
    dataset_name_to_task_id: Dict[str, int],
    get_task_variables,
    tree_agent,
    dataset_descriptions,
    prompting,
    metric_func_by_task,
    run_baselines: bool = True,
) -> pd.DataFrame:
    all_ds = [k for k in dataset_descriptions.desc.keys() if k != dataset_name]
    wrong_dataset_name = all_ds[0] if all_ds else None

    completed = _load_completed_keys(results_jsonl_path)
    print(f"Resuming: {len(completed)} results already exist")

    rows: List[Dict[str, Any]] = []

    def _append_and_save(row: Dict[str, Any]):
        rows.append(row)
        with open(results_jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    for repeat_index in range(n_repeats):
        task = openml.tasks.get_task(dataset_name_to_task_id[dataset_name])
        data = get_task_variables(task, fold=0, repeat=repeat_index)
        task_type = data["task_type"]
        metric_name = prompting.metrics_by_task[task_type]
        higher_is_better = _metric_higher_is_better(metric_name)
        metric_func = metric_func_by_task[task_type]

        X_full = data["X_train"].reset_index(drop=True)
        y_full = np.asarray(data["y_train"]).copy()
        X_test_clean = data["X_test"].reset_index(drop=True)
        y_test = np.asarray(data["y_test"]).copy()

        split_seed = _stable_seed(dataset_name, "split", repeat_index)
        X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            random_state=split_seed,
            stratify=y_full if task_type != "regression" else None,
        )

        for noise_level in noise_levels:
            corr_seed = _stable_seed(dataset_name, "corr", repeat_index, noise_level)
            corruptor = TrainValCorruptor(
                noise_level=noise_level, seed=corr_seed, cfg=corruption_cfg
            ).fit(X_train_base)

            X_train = corruptor.transform_X(X_train_base, split_name="train")
            X_val = corruptor.transform_X(X_val_base, split_name="val")
            y_train, y_val = corruptor.transform_y_train_val(
                y_train_base.copy(), y_val_base.copy()
            )
            X_test_corrupt = corruptor.transform_X(X_test_clean, split_name="test")

            # ─────────────────────────────────────────────────────────────
            # Baselines (CatBoost + Naive) — once per (repeat, noise_level)
            # ─────────────────────────────────────────────────────────────
            if run_baselines:
                base_row = dict(
                    dataset=dataset_name,
                    repeat=repeat_index,
                    task_type=task_type,
                    metric=metric_name,
                    higher_is_better=higher_is_better,
                    noise_level=noise_level,
                )

                # --- CatBoost ---
                cb_key = (repeat_index, noise_level, "catboost")
                if cb_key not in completed:
                    try:
                        score_A, score_B = _run_catboost(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test_clean,
                            X_test_corrupt,
                            y_test,
                            task_type,
                            metric_func,
                        )
                        row = {
                            **base_row,
                            "metadata_mode": "catboost",
                            "score_scopeA_clean_test": score_A,
                            "score_scopeB_corrupt_test": score_B,
                            "perf_scopeA": score_A if higher_is_better else -score_A,
                            "perf_scopeB": score_B if higher_is_better else -score_B,
                        }
                        _append_and_save(row)
                        print(
                            f"[r={repeat_index} nl={noise_level} meta=catboost] A={score_A:.5f}  B={score_B:.5f}"
                        )
                    except Exception as e:
                        print(
                            f"[r={repeat_index} nl={noise_level} meta=catboost] FAILED: {e}"
                        )

                # --- Naive baselines ---
                # For regression: just one baseline (mean)
                # For classification: both soft (class frequencies) and hard (majority class)
                if task_type == "regression":
                    baseline_variants = [("baseline", "soft")]
                else:
                    baseline_variants = [
                        ("baseline (soft)", "soft"),
                        ("baseline (hard)", "hard"),
                    ]

                for bl_name, bl_variant in baseline_variants:
                    bl_key = (repeat_index, noise_level, bl_name)
                    if bl_key not in completed:
                        try:
                            score_A, score_B = _run_baseline(
                                y_train,
                                y_test,
                                task_type,
                                metric_func,
                                variant=bl_variant,
                            )
                            row = {
                                **base_row,
                                "metadata_mode": bl_name,
                                "score_scopeA_clean_test": score_A,
                                "score_scopeB_corrupt_test": score_B,
                                "perf_scopeA": score_A
                                if higher_is_better
                                else -score_A,
                                "perf_scopeB": score_B
                                if higher_is_better
                                else -score_B,
                            }
                            _append_and_save(row)
                            print(
                                f"[r={repeat_index} nl={noise_level} meta={bl_name}] A={score_A:.5f}  B={score_B:.5f}"
                            )
                        except Exception as e:
                            print(
                                f"[r={repeat_index} nl={noise_level} meta={bl_name}] FAILED: {e}"
                            )

            # ─────────────────────────────────────────────────────────────
            # LLM agent runs (one per metadata_mode)
            # ─────────────────────────────────────────────────────────────
            for metadata_mode in metadata_modes:
                key = (repeat_index, noise_level, metadata_mode)
                if key in completed:
                    print(
                        f"[r={repeat_index} nl={noise_level} meta={metadata_mode}] SKIPPED (exists)"
                    )
                    continue

                prompt = make_prompt(
                    dataset_name=dataset_name,
                    task_type=task_type,
                    X_train=X_train,
                    metadata_mode=metadata_mode,
                    dataset_descriptions=dataset_descriptions,
                    prompting=prompting,
                    wrong_dataset_name=wrong_dataset_name,
                )

                attempt, done, result = 0, False, None
                while attempt < MAX_RETRIES and not done:
                    try:
                        result = tree_agent.TreeAgent(model=model).run(
                            task=prompt,
                            additional_args=dict(
                                X_train=X_train.copy(),
                                y_train=np.asarray(y_train).copy(),
                                X_val=X_val.copy(),
                                y_val=np.asarray(y_val).copy(),
                            ),
                        )
                        y_pred_A = result["model"].predict(
                            result["preprocess_features"](X_test_clean.copy())
                        )
                        y_pred_B = result["model"].predict(
                            result["preprocess_features"](X_test_corrupt.copy())
                        )

                        if task_type == "multiclass":
                            y_pred_A = _normalize_multiclass_proba(y_pred_A)
                            y_pred_B = _normalize_multiclass_proba(y_pred_B)

                        score_A = float(metric_func(y_test, y_pred_A))
                        score_B = float(metric_func(y_test, y_pred_B))
                        perf_A = score_A if higher_is_better else -score_A
                        perf_B = score_B if higher_is_better else -score_B

                        row = dict(
                            dataset=dataset_name,
                            repeat=repeat_index,
                            task_type=task_type,
                            metric=metric_name,
                            higher_is_better=higher_is_better,
                            noise_level=noise_level,
                            metadata_mode=metadata_mode,
                            score_scopeA_clean_test=score_A,
                            score_scopeB_corrupt_test=score_B,
                            perf_scopeA=perf_A,
                            perf_scopeB=perf_B,
                        )
                        _append_and_save(row)
                        print(
                            f"[r={repeat_index} nl={noise_level} meta={metadata_mode}] A={score_A:.5f}  B={score_B:.5f}"
                        )
                        done = True
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        wait_time = INITIAL_BACKOFF_SEC * (2**attempt)
                        print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                        print(f"Retrying in {wait_time:.0f}s...")
                        time.sleep(wait_time)
                        attempt += 1

                if result is None:
                    print(
                        f"[r={repeat_index} nl={noise_level} meta={metadata_mode}] FAILED after {MAX_RETRIES} attempts"
                    )

    return pd.DataFrame(rows)


# -----------------------
# Plot (scope A, curves=metadata)
# -----------------------
def plot_scopeA_curves(df: pd.DataFrame, title: str = "") -> None:
    # Use perf_scopeA so y is always "higher is better" even for losses.
    dfa = df.copy()

    # aggregate
    agg = dfa.groupby(["noise_level", "metadata_mode"], as_index=False).agg(
        mean_perf=("perf_scopeA", "mean"),
        std_perf=("perf_scopeA", "std"),
        n=("perf_scopeA", "count"),
    )
    agg["sem"] = agg["std_perf"] / np.sqrt(agg["n"].clip(lower=1))

    noise_levels = sorted(agg["noise_level"].unique())
    x = np.array(noise_levels)

    plt.figure()
    for meta in sorted(agg["metadata_mode"].unique()):
        sub = (
            agg[agg["metadata_mode"] == meta]
            .set_index("noise_level")
            .reindex(noise_levels)
        )
        plt.errorbar(
            x,
            sub["mean_perf"].to_numpy(),
            yerr=sub["sem"].to_numpy(),
            marker="o",
            capsize=3,
            label=meta,
        )

    plt.xticks(x, ["none", "mild", "some", "complete"])
    plt.xlabel("Noise level (train/val corruption; test is clean for scope A)")
    plt.ylabel("Performance (higher is better; losses negated)")
    plt.title(title or "Scope A: clean test performance vs noise level")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()
    dataset_name = args.dataset_name

    with open("tokens.txt") as file:
        tokens = file.read().splitlines()
    HF_TOKEN_HERE = tokens[0]
    API_TOKEN_HERE = tokens[1]
    API_ENDPOINT_HERE = tokens[2]
    MODEL = tokens[3]

    results_path = f"data/tree_scores.{dataset_name}.{MODEL}.json"

    login(token=HF_TOKEN_HERE)
    model = proxy_api_model.ProxyAPIModel(
        model_id=MODEL,
        api_base=API_ENDPOINT_HERE,
        api_key=API_TOKEN_HERE,
        max_new_tokens=1024 * 8,
    )

    # Load tabular benchmark
    tabarena_version = "tabarena-v0.1"
    benchmark_suite = openml.study.get_suite(tabarena_version)
    task_ids = benchmark_suite.tasks
    dataset_name_to_task_id = {}
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        n_samples = dataset.qualities["NumberOfInstances"]
        if n_samples < 2_500:
            dataset_name_to_task_id[dataset.name] = task_id
            print(dataset.name, int(n_samples), task_id)

    cfg = CorruptionConfig(
        mild_mask_rate=0.05,
        mild_noise_std=0.03,
        mild_scale_low=0.8,
        mild_scale_high=1.2,
        mild_rotate_frac=0.3,
        some_shuffle_col_frac=0.5,
    )

    df = run_leakage_sweep(
        dataset_name=dataset_name,
        model=model,
        n_repeats=2,
        noise_levels=[0, 1, 2, 3],
        metadata_modes=["true", "none", "wrong"],
        corruption_cfg=cfg,
        results_jsonl_path=results_path.replace(".json", ".jsonl"),
        openml=openml,
        dataset_name_to_task_id=dataset_name_to_task_id,
        get_task_variables=get_task_variables,
        tree_agent=tree_agent,
        dataset_descriptions=dataset_descriptions,
        prompting=prompting,
        metric_func_by_task=metric_func_by_task,
    )

    csv_path = f"data/tree_scores.{dataset_name}.{MODEL}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

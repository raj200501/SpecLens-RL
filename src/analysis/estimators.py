"""Metric estimators for the SpecLens-RL framework."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.utils.metrics_common import safe_corr


def hacking_incidence(df: pd.DataFrame) -> float:
    return float(df["reward_hack"].mean()) if not df.empty else float("nan")


def finite_policy_stress_check(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    grouped = df.groupby("method")["true_score"].mean()
    if grouped.empty:
        return float("nan")
    return float(grouped.max() - grouped.min())


def specification_gap_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "proxy_true_correlation": float(safe_corr(df["proxy_return"], df["true_score"])) if not df.empty else float("nan"),
        "hacking_incidence": hacking_incidence(df),
        "finite_policy_stress_check": finite_policy_stress_check(df),
    }


def tampering_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "tamper_utilization_rate": float("nan"),
            "tampering_benefit_ratio": float("nan"),
            "tampering_true_score_delta": float("nan"),
        }
    tampered = df[df["tamper_used"] == 1] if "tamper_used" in df.columns else pd.DataFrame()
    untampered = df[df["tamper_used"] == 0] if "tamper_used" in df.columns else pd.DataFrame()
    tamper_utilization_rate = float(df["tamper_used"].mean()) if "tamper_used" in df.columns else float("nan")
    tamper_proxy = tampered["proxy_return"].mean() if not tampered.empty else np.nan
    clean_proxy = untampered["proxy_return"].mean() if not untampered.empty else np.nan
    tampering_benefit_ratio = float(tamper_proxy / clean_proxy) if not np.isnan(clean_proxy) and clean_proxy != 0 else float("nan")
    tampering_true_score_delta = float(tampered["true_score"].mean() - untampered["true_score"].mean()) if not tampered.empty and not untampered.empty else float("nan")
    return {
        "tamper_utilization_rate": tamper_utilization_rate,
        "tampering_benefit_ratio": tampering_benefit_ratio,
        "tampering_true_score_delta": tampering_true_score_delta,
    }


def generalization_metrics(id_df: pd.DataFrame, ood_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "true_score_ID": float(id_df["true_score"].mean()) if not id_df.empty else float("nan"),
        "true_score_OOD": float(ood_df["true_score"].mean()) if not ood_df.empty else float("nan"),
        "hacking_ID": hacking_incidence(id_df),
        "hacking_OOD": hacking_incidence(ood_df),
        "generalization_gap_true": float(ood_df["true_score"].mean() - id_df["true_score"].mean()) if not id_df.empty and not ood_df.empty else float("nan"),
        "generalization_gap_hack": float(hacking_incidence(ood_df) - hacking_incidence(id_df)) if not id_df.empty and not ood_df.empty else float("nan"),
    }


def threshold_robustness(df: pd.DataFrame, thresholds: Iterable[float]) -> List[Dict[str, float]]:
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": float(threshold), "hacking_incidence": float((df["hack_score"] >= threshold).mean())})
    return rows


def summarize_main_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if episodes.empty:
        return pd.DataFrame(rows)
    for (env_name, method), group in episodes.groupby(["env_name", "method"]):
        id_df = group[group["split"] == "id"]
        ood_df = group[group["split"] == "ood"]
        spec = specification_gap_metrics(group[group["split"] == "id"])
        gen = generalization_metrics(id_df, ood_df)
        rows.append(
            {
                "env": env_name,
                "method": method,
                "proxy_true_correlation": spec["proxy_true_correlation"],
                "hacking_incidence_ID": gen["hacking_ID"],
                "hacking_incidence_OOD": gen["hacking_OOD"],
                "true_score_ID": gen["true_score_ID"],
                "true_score_OOD": gen["true_score_OOD"],
                "finite_policy_stress_check": spec["finite_policy_stress_check"],
            }
        )
    return pd.DataFrame(rows)


def summarize_seed_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if episodes.empty:
        return pd.DataFrame(rows)
    grouped = episodes.groupby(["env_name", "method", "seed", "split"], as_index=False)
    for (env_name, method, seed, split), group in grouped:
        rows.append(
            {
                "env": env_name,
                "method": method,
                "seed": int(seed),
                "split": str(split),
                "true_score": float(group["true_score"].mean()),
                "hack_rate": hacking_incidence(group),
                "proxy_return": float(group["proxy_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def seed_variability_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    seed_df = summarize_seed_metrics(episodes)
    if seed_df.empty:
        return pd.DataFrame()
    summary = (
        seed_df.groupby(["env", "method", "split"], as_index=False)
        .agg(
            true_score_mean=("true_score", "mean"),
            true_score_std=("true_score", "std"),
            hack_mean=("hack_rate", "mean"),
            hack_std=("hack_rate", "std"),
            proxy_return_mean=("proxy_return", "mean"),
            proxy_return_std=("proxy_return", "std"),
        )
        .fillna(0.0)
    )
    return summary


def seed_dominance_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    seed_df = summarize_seed_metrics(episodes)
    if seed_df.empty:
        return pd.DataFrame()
    rows = []
    for (env_name, split), group in seed_df.groupby(["env", "split"]):
        proxy_scores = group.loc[group["method"] == "PROXY_PPO", "true_score"]
        oracle_scores = group.loc[group["method"] == "ORACLE_TRUE_PPO", "true_score"]
        if proxy_scores.empty or oracle_scores.empty:
            continue
        proxy_max = float(proxy_scores.max())
        oracle_min = float(oracle_scores.min())
        rows.append(
            {
                "env": env_name,
                "split": str(split),
                "proxy_max_true_score": proxy_max,
                "oracle_min_true_score": oracle_min,
                "dominance_margin": oracle_min - proxy_max,
                "ordering_preserved": int(proxy_max < oracle_min),
            }
        )
    return pd.DataFrame(rows)

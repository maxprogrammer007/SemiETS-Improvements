import json
import glob
import os
import numpy as np
from collections import defaultdict


# ----------------------------
# CONFIG
# ----------------------------
PROPOSED_LOG_DIR = "experiment_logs"
BASELINE_LOG_DIR = "experiment_logs_baseline"
T_D = 0.5   # detection threshold used in baseline


def load_logs(log_dir):
    all_logs = []
    for file in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        with open(file, "r") as f:
            logs = json.load(f)
            all_logs.extend(logs)
    return all_logs


def summarize_logs(logs, mode="proposed"):
    stats = defaultdict(list)

    for entry in logs:
        det_conf = entry["det_conf"]
        loss = entry["loss"]

        stats["loss"].append(loss)

        if mode == "baseline":
            stats["used"].append(1 if entry["baseline_accept"] else 0)
        else:
            stats["used"].append(1 if entry["final_weight"] > 0 else 0)
            stats["weights"].append(entry["final_weight"])

        if "word_length" in entry:
            stats["word_length"].append(entry["word_length"])
            stats["low_conf"].append(1 if det_conf < T_D else 0)

    return stats


def print_summary(stats, mode):
    print("\n" + "=" * 50)
    print(f"SUMMARY ({mode.upper()})")
    print("=" * 50)

    used_ratio = np.mean(stats["used"])
    loss_mean = np.mean(stats["loss"])
    loss_std = np.std(stats["loss"])

    print(f"Sample utilization ratio     : {used_ratio:.3f}")
    print(f"Mean loss                    : {loss_mean:.4f}")
    print(f"Loss standard deviation      : {loss_std:.4f}")

    if mode == "proposed":
        print(f"Mean reliability weight      : {np.mean(stats['weights']):.4f}")
        print(f"Std reliability weight       : {np.std(stats['weights']):.4f}")

    if "word_length" in stats:
        short_words = [
            u for u, w in zip(stats["used"], stats["word_length"]) if w <= 3
        ]
        if short_words:
            print(f"Utilization (short words ≤3): {np.mean(short_words):.3f}")


def recovery_analysis(baseline_logs, proposed_logs):
    """
    How many samples rejected by baseline
    but used by proposed?
    """
    recovered = 0
    rejected = 0

    for b, p in zip(baseline_logs, proposed_logs):
        if not b["baseline_accept"]:
            rejected += 1
            if p["final_weight"] > 0:
                recovered += 1

    if rejected > 0:
        print("\nRecovered failure samples:")
        print(f"Recovered {recovered}/{rejected} "
              f"({recovered/rejected:.2%})")
    else:
        print("\nNo rejected samples found in baseline.")


def main():
    print("\nLoading logs...")

    baseline_logs = load_logs(BASELINE_LOG_DIR)
    proposed_logs = load_logs(PROPOSED_LOG_DIR)

    assert len(baseline_logs) == len(proposed_logs), \
        "Baseline and proposed logs must have same length"

    baseline_stats = summarize_logs(baseline_logs, mode="baseline")
    proposed_stats = summarize_logs(proposed_logs, mode="proposed")

    print_summary(baseline_stats, "baseline")
    print_summary(proposed_stats, "proposed")

    recovery_analysis(baseline_logs, proposed_logs)


if __name__ == "__main__":
    main()

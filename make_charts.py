"""
make_charts.py
==============
Generates static PNG charts into images/ for embedding directly in README.md.

    python make_charts.py

Why this exists: GitHub renders README.md natively but shows raw source for
.html files, so the interactive dashboard needs GitHub Pages to be viewable.
These PNGs give anyone landing on the repo page the headline visuals
immediately, with no clicks and no tooling.

Requires: pandas, matplotlib, scikit-learn
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("updated_data.xlsx")
OUT_DIR = Path("images")
TARGET, FAIL = "waec", "FAIL"
SEED = 42

C_HIGH = "#C2410C"
C_LOW = "#A8B0BA"
C_INK = "#1F2933"
C_GREY = "#7B8794"
C_LINE = "#E4E7EB"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "axes.edgecolor": C_LINE,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": C_LINE,
    "grid.linewidth": .9,
    "xtick.color": C_GREY,
    "ytick.color": C_GREY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH.resolve()} not found. Run `python build_dataset.py` first.")
    df = pd.read_excel(DATA_PATH)
    df["is_fail"] = (df[TARGET] == FAIL).astype(int)
    return df


def run_models(df):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, average_precision_score

    grade_cols = [c for c in df.columns
                  if c.endswith(("_term1", "_term2", "_term3", "_avg"))]
    avg_cols = [c for c in df.columns if c.endswith("_avg")]
    exclude = {"student_id", "parent_id", "score_band", TARGET, "is_fail"}
    full = [c for c in df.columns if c not in exclude and c not in avg_cols]
    early = [c for c in df.columns if c not in exclude and c not in grade_cols]

    def prep(feats):
        num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in feats if c not in num]
        return ColumnTransformer([
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("sc", StandardScaler(), num)])

    models = {
        "Logistic\nRegression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random\nForest": RandomForestClassifier(
            class_weight="balanced", random_state=SEED),
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
            ensemble=False),
    }

    out, coefs = {}, None
    for fs_name, feats in (("full", full), ("early", early)):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feats], df[TARGET], test_size=.30, random_state=SEED, stratify=df[TARGET])
        y_bin = (y_te == FAIL).astype(int)

        d = Pipeline([("p", prep(feats)), ("c", DummyClassifier(strategy="most_frequent"))])
        d.fit(X_tr, y_tr)
        out[(fs_name, "Baseline")] = (accuracy_score(y_te, d.predict(X_te)), y_bin.mean())

        for m_name, est in models.items():
            pipe = Pipeline([("p", prep(feats)), ("c", est)])
            pipe.fit(X_tr, y_tr)
            i = list(pipe.classes_).index(FAIL)
            s = pipe.predict_proba(X_te)[:, i]
            out[(fs_name, m_name)] = (accuracy_score(y_te, pipe.predict(X_te)),
                                      average_precision_score(y_bin, s))
            if fs_name == "early" and m_name.startswith("Logistic"):
                names = pipe.named_steps["p"].get_feature_names_out()
                coefs = pd.Series(pipe.named_steps["c"].coef_.ravel(),
                                  index=[n.split("__", 1)[-1] for n in names])
    return out, coefs


def chart_leakage(res):
    order = ["Baseline", "Logistic\nRegression", "Random\nForest", "SVM"]
    early = [res[("early", m)][0] for m in order]
    full = [res[("full", m)][0] for m in order]
    x = np.arange(len(order)); w = .36

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b1 = ax.bar(x - w/2, early, w, color=C_LOW, label="early (no grades)", zorder=3)
    b2 = ax.bar(x + w/2, full, w, color=C_HIGH, label="full (grades included)", zorder=3)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", fontsize=9, padding=3, color=C_INK)

    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10, color=C_INK)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    ax.set_ylabel("Accuracy on held-out test set")
    ax.set_axisbelow(True); ax.xaxis.grid(False)
    ax.set_title("The same pipeline, trained twice", fontsize=13, fontweight="bold",
                 color=C_INK, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2,
              bbox_to_anchor=(0, 1.02))
    gap = (max(full) - max(early)) * 100
    ax.text(.99, .04, f"~{gap:.0f} accuracy points come from leakage",
            transform=ax.transAxes, ha="right", fontsize=9.5, style="italic", color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "leakage_gap.png", bbox_inches="tight")
    plt.close(fig)


def chart_failures(df):
    base = df["is_fail"].mean()
    g = df.groupby("failures")["is_fail"].agg(rate="mean", n="size").reset_index()
    colors = [C_HIGH if r > base else C_LOW for r in g["rate"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(g["failures"].astype(int).astype(str), g["rate"], color=colors,
                  width=.6, zorder=3)
    ax.bar_label(bars, labels=[f"{r*100:.0f}%" for r in g["rate"]],
                 fontsize=10, padding=3, color=C_INK)
    ax.axhline(base, ls="--", color=C_INK, alpha=.45, lw=1.4, zorder=2)
    ax.text(.995, base + .022, f"cohort average {base*100:.1f}%", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=9, color=C_GREY)

    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{int(v)}\nn={int(n)}" for v, n in zip(g["failures"], g["n"])],
                       fontsize=10, color=C_INK)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    ax.set_xlabel("Prior class failures", labelpad=8)
    ax.set_ylabel("Fail rate")
    ax.set_axisbelow(True); ax.xaxis.grid(False)
    ax.set_title("Prior failures are the strongest signal", fontsize=13,
                 fontweight="bold", color=C_INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prior_failures.png", bbox_inches="tight")
    plt.close(fig)


def chart_coefficients(coefs):
    top = pd.concat([coefs.sort_values().head(8), coefs.sort_values().tail(8)])
    colors = [C_HIGH if v < 0 else C_LOW for v in top.values]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.barh(range(len(top)), top.values, color=colors, height=.72, zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9.5, color=C_INK)
    ax.axvline(0, color=C_INK, lw=1, alpha=.6)
    ax.set_xlabel("pushes toward FAIL  <-              ->  pushes toward PASS", labelpad=8)
    ax.set_axisbelow(True); ax.yaxis.grid(False)
    ax.set_title("What drives risk, honest features only", fontsize=13,
                 fontweight="bold", color=C_INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_drivers.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load()
    res, coefs = run_models(df)
    chart_leakage(res)
    chart_failures(df)
    chart_coefficients(coefs)
    print(f"Wrote 3 charts to {OUT_DIR.resolve()}/")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

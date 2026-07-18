"""
build_dataset.py
================
Reproducibly builds `updated_data.xlsx` from the ORIGINAL public UCI Student
Performance dataset (Cortez & Silva, 2008; CC BY 4.0), so the data lineage is
fully transparent and re-runnable.

Chain:  original UCI (Math + Portuguese)  ->  merge on shared attributes (~382
students)  ->  rename to the project schema  ->  map real grades + generate
clearly-labelled synthetic subjects  ->  derive target  ->  write updated_data.xlsx

Provenance / honesty notes
--------------------------
* `maths_*`   = REAL grades from the Mathematics file (G1/G2/G3).
* `english_*` = REAL grades from the Portuguese file (G1/G2/G3), relabelled.
* `biology_*`, `yoruba_*` = SYNTHETIC (seeded). They do NOT correspond to any
  real measurements. Set INCLUDE_SYNTHETIC_SUBJECTS = False to omit them — the
  model notebook detects grade columns dynamically, so it works with 2 or 4
  subjects without any edits.
* This does NOT reproduce the exact values of the original file (the first
  synthesis used an unknown random process); it reproduces the schema and the
  real-grade columns deterministically, which is the stronger lineage story.

Usage:  python build_dataset.py
Requires: pandas, numpy, openpyxl   (and internet, or local source CSVs)
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
OUTPUT_PATH  = Path("updated_data.xlsx")
RANDOM_SEED  = 42
PASS_THRESHOLD = 10          # 0-20 scale; >= is PASS. Tune to taste.
INCLUDE_SYNTHETIC_SUBJECTS = True   # False -> only real maths + english

# Source: the two real UCI CSVs (semicolon-delimited), served from a stable GitHub
# mirror because UCI's own legacy CSV URLs now 404 after their site redesign.
# These are the unmodified Cortez & Silva files (student-mat.csv / student-por.csv).
# If these ever move: download student.zip from
# https://archive.ics.uci.edu/dataset/320/student+performance , unzip, and drop
# student-mat.csv + student-por.csv next to this script (auto-detected below).
_GH = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/"
URL_MAT = _GH + "student-mat.csv"
URL_POR = _GH + "student-por.csv"
LOCAL_MAT = Path("student-mat.csv")   # used if present (offline / URL moved)
LOCAL_POR = Path("student-por.csv")

# Official merge keys (Cortez's student-merge.R) -> ~382 students in both files.
MERGE_KEYS = ["school", "sex", "age", "address", "famsize", "Pstatus",
              "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

# original UCI col -> project col  (non-key attributes; math side is used)
RENAME = {
    "Pstatus": "parent_marital_status", "Medu": "m_education", "Fedu": "f_education",
    "Mjob": "m_job", "Fjob": "f_job", "traveltime": "travel_time",
    "studytime": "study_time", "goout": "goout_friends", "romantic": "romantic_rel",
    "health": "health_status", "famsup": "family_support", "paid": "extra_lessons",
    "activities": "extra_curricular", "internet": "internet_at_home",
}

FINAL_COLUMNS = [
    "student_id", "sex", "age", "address", "famsize", "reason", "guardian",
    "romantic_rel", "health_status", "absences", "travel_time", "study_time",
    "freetime", "goout_friends", "school", "parent_id", "parent_marital_status",
    "m_education", "f_education", "m_job", "f_job", "family_support",
    "extra_lessons", "extra_curricular", "internet_at_home",
    "maths_term1", "maths_term2", "maths_term3", "maths_avg",
    "english_term1", "english_term2", "english_term3", "english_avg",
    "biology_term1", "biology_term2", "biology_term3", "biology_avg",
    "yoruba_term1", "yoruba_term2", "yoruba_term3", "yoruba_avg",
    "failures", "score_band", "waec",
]


def load_sources():
    if LOCAL_MAT.exists() and LOCAL_POR.exists():
        print(f"Loading local source files: {LOCAL_MAT}, {LOCAL_POR}")
        mat = pd.read_csv(LOCAL_MAT, sep=";")
        por = pd.read_csv(LOCAL_POR, sep=";")
    else:
        print("Downloading original UCI Student Performance data (Math + Portuguese)...")
        mat = pd.read_csv(URL_MAT, sep=";")
        por = pd.read_csv(URL_POR, sep=";")
    print(f"  math file:       {mat.shape}")
    print(f"  portuguese file: {por.shape}")
    return mat, por


def synth_terms(base_signal, rng, n_terms=3, lo=0, hi=20):
    """Seeded synthetic term grades, correlated with a base ability signal.
    Clearly NOT real data — see module docstring."""
    terms = []
    for _ in range(n_terms):
        vals = np.clip(np.round(base_signal + rng.normal(0, 3, len(base_signal))), lo, hi)
        terms.append(vals.astype(int))
    return terms


def build():
    rng = np.random.default_rng(RANDOM_SEED)
    mat, por = load_sources()

    # Merge the two real subjects -> students present in both (~382).
    merged = pd.merge(mat, por, on=MERGE_KEYS, suffixes=("_mat", "_por"))
    merged = merged.reset_index(drop=True)
    print(f"Merged (students in both subjects): {merged.shape[0]} rows")

    out = pd.DataFrame(index=merged.index)

    # Identifiers (synthetic, deterministic)
    out["student_id"] = [f"STU{i:04d}" for i in range(len(merged))]
    out["parent_id"]  = [f"PAR{i:04d}" for i in range(len(merged))]

    # Merge-key columns (single, unsuffixed)
    out["school"] = merged["school"]
    out["sex"] = merged["sex"]
    out["age"] = merged["age"]
    out["address"] = merged["address"]
    out["famsize"] = merged["famsize"]
    out["reason"] = merged["reason"]
    out["parent_marital_status"] = merged["Pstatus"]
    out["m_education"] = merged["Medu"]
    out["f_education"] = merged["Fedu"]
    out["m_job"] = merged["Mjob"]
    out["f_job"] = merged["Fjob"]
    out["internet_at_home"] = merged["internet"]

    # Non-key attributes: take the math-side value (suffix _mat)
    def side(col): return merged[f"{col}_mat"]
    out["guardian"] = side("guardian")
    out["romantic_rel"] = side("romantic")
    out["health_status"] = side("health")
    out["absences"] = side("absences")
    out["travel_time"] = side("traveltime")
    out["study_time"] = side("studytime")
    out["freetime"] = side("freetime")
    out["goout_friends"] = side("goout")
    out["family_support"] = side("famsup")
    out["extra_lessons"] = side("paid")
    out["extra_curricular"] = side("activities")
    out["failures"] = side("failures")

    # REAL grades: maths <- Math G1/G2/G3, english <- Portuguese G1/G2/G3
    out["maths_term1"] = merged["G1_mat"]; out["maths_term2"] = merged["G2_mat"]; out["maths_term3"] = merged["G3_mat"]
    out["maths_avg"] = out[["maths_term1", "maths_term2", "maths_term3"]].mean(axis=1).round(1)
    out["english_term1"] = merged["G1_por"]; out["english_term2"] = merged["G2_por"]; out["english_term3"] = merged["G3_por"]
    out["english_avg"] = out[["english_term1", "english_term2", "english_term3"]].mean(axis=1).round(1)

    # SYNTHETIC subjects (seeded), correlated with real ability. Clearly not real.
    real_ability = out[["maths_avg", "english_avg"]].mean(axis=1).to_numpy()
    if INCLUDE_SYNTHETIC_SUBJECTS:
        for subj in ["biology", "yoruba"]:
            t1, t2, t3 = synth_terms(real_ability, rng)
            out[f"{subj}_term1"], out[f"{subj}_term2"], out[f"{subj}_term3"] = t1, t2, t3
            out[f"{subj}_avg"] = np.round((t1 + t2 + t3) / 3, 1)

    # Target + band, derived from REAL subjects only (keeps target off synthetic noise)
    overall_real = out[["maths_avg", "english_avg"]].mean(axis=1)
    out["waec"] = np.where(overall_real >= PASS_THRESHOLD, "PASS", "FAIL")
    out["score_band"] = pd.cut(overall_real, bins=[-0.1, 8, 12, 20],
                               labels=["low", "mid", "high"]).astype(str)

    # Order columns (drop synthetic subject cols from the target list if omitted)
    cols = [c for c in FINAL_COLUMNS
            if c in out.columns or not c.startswith(("biology_", "yoruba_"))]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    out.to_excel(OUTPUT_PATH, index=False)

    # Summary / provenance print
    real = [c for c in out.columns if c.startswith(("maths_", "english_"))]
    synth = [c for c in out.columns if c.startswith(("biology_", "yoruba_"))]
    print(f"\nWrote {OUTPUT_PATH}  ->  {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"Target balance: {dict(out['waec'].value_counts())}")
    print(f"REAL grade cols   : {real}")
    print(f"SYNTHETIC grade cols: {synth if synth else '(none)'}")
    return out


if __name__ == "__main__":
    build()
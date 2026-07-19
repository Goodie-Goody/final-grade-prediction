"""
dashboard.py
============
Builds a self-contained two page dashboard (index.html) from
updated_data.xlsx. No Power BI, no server, no account.

    python dashboard.py        # writes index.html, open it in any browser

Page 1, Risk Factors: which student characteristics are associated with
failure, presented as fail rates by category.

Page 2, The Leakage Story: the same pipeline trained twice, once with the
term grades included and once without, showing how much of the apparent
accuracy was an artefact of the target leaking into the features.

Design notes:
  * Colour encodes meaning, never decoration. Terracotta means above the
    cohort fail rate, slate grey means below.
  * Grey and terracotta differ in saturation as well as hue, so they stay
    separable under deuteranopia, protanopia and tritanopia. Colour is never
    the only channel: every bar carries a numeric label and every category
    label carries its sample size.
  * Fail rate by category, not average score split by pass or fail. The pass
    or fail flag is derived from the grades, so the latter restates arithmetic.

Requires: pandas, plotly, scikit-learn
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA_PATH = Path("updated_data.xlsx")
OUT_PATH = Path("index.html")
TARGET, FAIL = "waec", "FAIL"
MIN_N = 10
SEED = 42

C_HIGH = "#C2410C"
C_LOW = "#A8B0BA"
C_INK = "#1F2933"
C_GREY = "#7B8794"
C_LINE = "#E4E7EB"

TRAVEL = {1: "<15 min", 2: "15-30 min", 3: "30-60 min", 4: ">1 hour"}
STUDY = {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}
EDU = {0: "None", 1: "Primary", 2: "Middle", 3: "Secondary", 4: "Higher"}

PLOT_CFG = {"displaylogo": False, "responsive": True,
            "displayModeBar": False, "scrollZoom": False,
            "doubleClick": False, "staticPlot": False}
FONT = dict(family="Segoe UI, Helvetica, Arial", color=C_INK, size=12)


# ----------------------------------------------------------------- data
def load():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH.resolve()} not found. Run `python build_dataset.py` first.")
    df = pd.read_excel(DATA_PATH)
    df["is_fail"] = (df[TARGET] == FAIL).astype(int)
    df["travel_band"] = df["travel_time"].map(TRAVEL)
    df["study_band"] = df["study_time"].map(STUDY)
    df["m_edu_band"] = df["m_education"].map(EDU)
    df["f_edu_band"] = df["f_education"].map(EDU)
    return df


def fail_rate(df, col, order=None):
    g = df.groupby(col)["is_fail"].agg(rate="mean", n="size").reset_index()
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    return g.dropna(subset=[col])


# ------------------------------------------------------- page 1 figures
def rate_fig(df, col, base, order=None, height=330, title=None):
    g = fail_rate(df, col, order)
    ticks = [f"{c}<br><span style='font-size:9.5px;color:#9AA5B1'>n={int(n)}</span>"
             for c, n in zip(g[col].astype(str), g["n"])]
    colors = [C_HIGH if r > base else C_LOW for r in g["rate"]]
    patterns = ["/" if n < MIN_N else "" for n in g["n"]]
    labels = [f"{r*100:.0f}%" + ("*" if n < MIN_N else "")
              for r, n in zip(g["rate"], g["n"])]
    hover = [f"<b>{c}</b><br>fail rate: {r*100:.1f}%<br>students: {int(n)}"
             + ("<br><i>small sample, unreliable</i>" if n < MIN_N else "")
             for c, r, n in zip(g[col].astype(str), g["rate"], g["n"])]

    fig = go.Figure(go.Bar(
        x=ticks, y=g["rate"], marker_color=colors, marker_pattern_shape=patterns,
        marker_line=dict(width=0), marker_cornerradius=5,
        text=labels, textposition="outside", textfont=dict(size=12),
        hovertext=hover, hoverinfo="text"))
    fig.add_hline(y=base, line_dash="dash", line_color=C_INK, opacity=.45, line_width=1.5)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.98], gridcolor=C_LINE,
                     zeroline=False, fixedrange=True)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11.5), fixedrange=True,
                     automargin=True)
    fig.update_layout(height=height, template="plotly_white", bargap=0.42, font=FONT,
                      margin=dict(t=30 if title else 12, b=40, l=46, r=14),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False, showlegend=False, 
                      title=dict(text=title, font=dict(size=13, color=C_GREY),
                                 x=0, xanchor="left") if title else None)
    return fig


def support_fig(df, base, height=340):
    flags = {"family_support": "Family support", "internet_at_home": "Internet at home",
             "extra_lessons": "Extra lessons", "extra_curricular": "Extra-curricular"}
    fig = go.Figure()
    for val in ("yes", "no"):
        ys, xs, txt, cols, hov = [], [], [], [], []
        for c, label in flags.items():
            sub = df[df[c] == val]
            if not len(sub):
                continue
            r = sub["is_fail"].mean()
            ys.append(label); xs.append(r)
            txt.append(f"  {val}: {r*100:.0f}%")
            cols.append(C_HIGH if r > base else C_LOW)
            hov.append(f"<b>{label} = {val}</b><br>fail rate: {r*100:.1f}%"
                       f"<br>students: {len(sub)}")
        fig.add_trace(go.Bar(y=ys, x=xs, orientation="h", marker_color=cols,
                             marker_line=dict(width=0), marker_cornerradius=5,
                             text=txt, textposition="auto",
                             textfont=dict(size=11, color="white"),
                             insidetextanchor="middle",
                             hovertext=hov, hoverinfo="text"))
    fig.add_vline(x=base, line_dash="dash", line_color=C_INK, opacity=.45, line_width=1.5)
    fig.update_xaxes(tickformat=".0%", range=[0, 0.72], gridcolor=C_LINE, fixedrange=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11.5),
                     fixedrange=True, automargin=True)
    fig.update_layout(height=height, template="plotly_white", bargap=0.45, barmode="group",
                      font=FONT, margin=dict(t=12, b=40, l=118, r=22),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False, showlegend=False)
    return fig


# ------------------------------------------------------- page 2 models
def run_models(df):
    """Train the same pipeline on both feature sets. Returns metrics + PR curves."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, recall_score,
                                 average_precision_score, precision_recall_curve)

    grade_cols = [c for c in df.columns
                  if c.endswith(("_term1", "_term2", "_term3", "_avg"))]
    avg_cols = [c for c in df.columns if c.endswith("_avg")]
    exclude = {"student_id", "parent_id", "score_band", TARGET, "is_fail",
               "travel_band", "study_band", "m_edu_band", "f_edu_band"}
    full = [c for c in df.columns if c not in exclude and c not in avg_cols]
    early = [c for c in df.columns if c not in exclude and c not in grade_cols]

    def prep(feats):
        num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        cat = [c for c in feats if c not in num]
        return ColumnTransformer([
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("sc", StandardScaler(), num)])

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=SEED),
        # CalibratedClassifierCV rather than SVC(probability=True), which is
        # deprecated in scikit-learn 1.9 and removed in 1.11.
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
            ensemble=False),
    }

    rows, curves = [], {}
    for fs_name, feats in (("full", full), ("early", early)):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[feats], df[TARGET], test_size=.30, random_state=SEED, stratify=df[TARGET])
        y_bin = (y_te == FAIL).astype(int)

        dummy = Pipeline([("p", prep(feats)), ("c", DummyClassifier(strategy="most_frequent"))])
        dummy.fit(X_tr, y_tr)
        rows.append(dict(feature_set=fs_name, model="Baseline",
                         accuracy=accuracy_score(y_te, dummy.predict(X_te)),
                         recall=recall_score(y_te, dummy.predict(X_te),
                                             pos_label=FAIL, zero_division=0),
                         pr_auc=y_bin.mean()))

        for m_name, est in models.items():
            pipe = Pipeline([("p", prep(feats)), ("c", est)])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            i = list(pipe.classes_).index(FAIL)
            s = pipe.predict_proba(X_te)[:, i]
            rows.append(dict(feature_set=fs_name, model=m_name,
                             accuracy=accuracy_score(y_te, pred),
                             recall=recall_score(y_te, pred, pos_label=FAIL, zero_division=0),
                             pr_auc=average_precision_score(y_bin, s)))
            p, r, _ = precision_recall_curve(y_bin, s)
            curves[(fs_name, m_name)] = (r, p)

    return pd.DataFrame(rows), curves, y_bin.mean(), len(full), len(early)


def leakage_fig(res, height=400):
    order = ["Baseline", "Logistic Regression", "Random Forest", "SVM"]
    fig = go.Figure()
    for fs, color, name in (("early", C_LOW, "early (no grades)"),
                            ("full", C_HIGH, "full (grades included)")):
        sub = res[res.feature_set == fs].set_index("model").loc[order]
        fig.add_trace(go.Bar(
            x=order, y=sub["accuracy"], name=name, marker_color=color,
            marker_line=dict(width=0), marker_cornerradius=5,
            text=[f"{v:.2f}" for v in sub["accuracy"]], textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>" + name + "<br>accuracy: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.12], gridcolor=C_LINE, zeroline=False,
                     tickformat=".0%", fixedrange=True)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12), fixedrange=True,
                     automargin=True)
    fig.update_layout(height=height, template="plotly_white", barmode="group",
                      bargap=0.42, font=FONT, margin=dict(t=16, b=44, l=48, r=14),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
                      legend=dict(orientation="h", y=1.12, x=0, title_text=""))
    return fig


def pr_fig(curves, prevalence, height=400):
    fig = go.Figure()
    styles = {("full", "SVM"): (C_HIGH, "solid", "full, SVM"),
              ("full", "Logistic Regression"): (C_HIGH, "dot", "full, Logistic Regression"),
              ("early", "Random Forest"): (C_LOW, "solid", "early, Random Forest"),
              ("early", "Logistic Regression"): (C_LOW, "dot", "early, Logistic Regression")}
    for key, (color, dash, label) in styles.items():
        if key not in curves:
            continue
        r, p = curves[key]
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=label,
                                 line=dict(color=color, width=2.4, dash=dash),
                                 hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>"))
    fig.add_hline(y=prevalence, line_dash="dash", line_color=C_INK, opacity=.45,
                  line_width=1.5,
                  annotation_text=f"chance ({prevalence:.2f})",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=C_GREY))
    fig.update_xaxes(title="Recall (share of failing students caught)", range=[0, 1],
                     gridcolor=C_LINE, title_font=dict(size=11.5, color=C_GREY),
                     fixedrange=True)
    fig.update_yaxes(title="Precision", range=[0, 1.03], gridcolor=C_LINE,
                     title_font=dict(size=11.5, color=C_GREY), fixedrange=True)
    fig.update_layout(height=height, template="plotly_white", font=FONT,
                      margin=dict(t=16, b=52, l=52, r=14),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
                      legend=dict(orientation="h", y=-0.22, x=0, title_text="",
                                  font=dict(size=11)))
    return fig


# ---------------------------------------------------------------- html
def h(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CFG)


def section(num, title, lede, body):
    return f"""
<section class="sec">
  <div class="rail"><span class="num">{num}</span></div>
  <div class="secbody">
    <div class="sectext"><h3>{title}</h3><p>{lede}</p></div>
    <div class="secvis">{body}</div>
  </div>
</section>"""


def kpis(df, base):
    cards = [("Students", f"{len(df):,}", "merged Maths and Portuguese cohort"),
             ("Overall fail rate", f"{base*100:.1f}%", f"{int(df.is_fail.sum())} of {len(df)}"),
             ("Prior class failures", f"{int(df.failures.sum()):,}",
              "sum of past failures, not students"),
             ("Mean maths average", f"{df.maths_avg.mean():.1f} / 20", "real UCI grades")]
    return '<div class="kpis">' + "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in cards) + "</div>"


def build_html(df):
    base = df.is_fail.mean()
    res, curves, prevalence, n_full, n_early = run_models(df)

    def get(fs, m, col):
        return res[(res.feature_set == fs) & (res.model == m)][col].iloc[0]

    MODELS = ("Logistic Regression", "Random Forest", "SVM")
    full_acc = max(get("full", m, "accuracy") for m in MODELS)
    early_acc = max(get("early", m, "accuracy") for m in MODELS)
    full_pr = max(get("full", m, "pr_auc") for m in MODELS)
    early_pr = max(get("early", m, "pr_auc") for m in MODELS)
    base_acc = get("early", "Baseline", "accuracy")
    lr_acc = get("early", "Logistic Regression", "accuracy")
    lr_rec = get("early", "Logistic Regression", "recall")

    # ---- page 1 sections
    p1 = kpis(df, base) + f"""
<div class="legendbar">
  <span><span class="sw" style="background:{C_HIGH}"></span>Above the cohort fail rate ({base*100:.1f}%)</span>
  <span><span class="sw" style="background:{C_LOW}"></span>Below it</span>
  <span><span class="sw" style="background:repeating-linear-gradient(45deg,{C_HIGH},{C_HIGH} 3px,#fff 3px,#fff 5px)"></span>Hatched or starred: fewer than {MIN_N} students</span>
  <span>Dashed line: cohort average</span>
</div>"""

    p1 += section(1, "Prior failures dominate everything else",
                  "A student who has already failed a class is the clearest signal in the "
                  "dataset. The fail rate climbs steeply with each prior failure, and this "
                  "matches the modelling, where this feature carried by far the largest "
                  "coefficient.",
                  h(rate_fig(df, "failures", base, height=340)))

    p1 += section(2, "Study time helps, then stops helping",
                  "Fail rates fall as weekly study time rises, but the gain flattens after "
                  "the five to ten hour band. More hours beyond that point are not "
                  "associated with better outcomes. Longer commutes coincide with higher "
                  "risk, though those bands hold very few students.",
                  '<div class="twoup">' + h(rate_fig(df, "study_band", base, list(STUDY.values()),
                                                     title="Weekly study time"))
                  + h(rate_fig(df, "travel_band", base, list(TRAVEL.values()),
                               title="Travel time to school")) + "</div>")

    p1 += section(3, "Family background shows a real gradient",
                  "Maternal education tracks fail rates clearly, from primary education down "
                  "to higher education. The paternal gradient runs the same way but is "
                  "weaker and noisier at the extremes.",
                  '<div class="twoup">' + h(rate_fig(df, "m_edu_band", base, list(EDU.values()),
                                                     title="Mother's education"))
                  + h(rate_fig(df, "f_edu_band", base, list(EDU.values()),
                               title="Father's education")) + "</div>")

    p1 += section(4, "Support factors move the needle less than expected",
                  "Extra lessons, internet access and extra-curricular activities each shift "
                  "the fail rate by only a few points. Students receiving family support fail "
                  "at a slightly higher rate, which most likely means support is directed "
                  "towards students who are already struggling rather than causing the "
                  "struggle.",
                  h(support_fig(df, base)))

    # ---- page 2 sections
    p2 = f"""
<div class="kpis">
  <div class="card"><div class="k">With grades included</div><div class="v">{full_acc:.2f}</div>
    <div class="s">accuracy, and almost meaningless</div></div>
  <div class="card"><div class="k">Without grades</div><div class="v">{early_acc:.2f}</div>
    <div class="s">accuracy, and the honest number</div></div>
  <div class="card"><div class="k">Majority baseline</div><div class="v">{base_acc:.2f}</div>
    <div class="s">always predict pass</div></div>
  <div class="card"><div class="k">Difference</div><div class="v">{(full_acc-early_acc)*100:.0f} pts</div>
    <div class="s">accuracy attributable to leakage</div></div>
</div>"""

    p2 += section(1, "The target was hiding inside the features",
                  "The pass or fail label in this dataset is derived from the subject grades. "
                  "Any model handed those grades is partly being given the answer it is "
                  "supposed to predict. It is also not an early warning in any useful sense, "
                  "because third term grades arrive at roughly the same time as the exam "
                  "result, once the window to intervene has closed.",
                  f"""
<div class="timeline">
  <div class="tl-row">
    <div class="tl-label">Known at enrolment</div>
    <div class="tl-items"><span class="ok">demographics</span><span class="ok">parental education</span>
      <span class="ok">travel time</span><span class="ok">prior failures</span></div>
  </div>
  <div class="tl-row">
    <div class="tl-label">Known during the year</div>
    <div class="tl-items"><span class="ok">study time</span><span class="ok">absences</span>
      <span class="ok">support factors</span></div>
  </div>
  <div class="tl-row">
    <div class="tl-label">Known only at the end</div>
    <div class="tl-items"><span class="bad">term 1, 2, 3 grades</span><span class="bad">subject averages</span>
      <span class="bad">score band</span></div>
  </div>
  <div class="tl-note">The <b>early</b> feature set uses only the first two rows
    ({n_early} features). The <b>full</b> set adds the third ({n_full} features).</div>
</div>""")

    p2 += section(2, "The same pipeline, trained twice",
                  "Nothing changes between these two runs except which columns the model is "
                  "allowed to see. The grade features lift accuracy by roughly "
                  f"{(full_acc-early_acc)*100:.0f} points, and that entire lift is an "
                  "artefact rather than predictive skill.",
                  h(leakage_fig(res)))

    p2 += section(3, "Ranking at-risk students is where the real signal shows",
                  "Precision and recall matter more than accuracy when the class of interest "
                  "is the minority. The honest models sit well above chance at "
                  f"{early_pr:.2f} against {prevalence:.2f}, roughly "
                  f"{early_pr/prevalence:.1f} times better than guessing, while the leaky "
                  f"models reach {full_pr:.2f} for reasons that will not generalise.",
                  h(pr_fig(curves, prevalence)))

    p2 += section(4, "Why accuracy on its own would have misled you",
                  f"Logistic regression on the honest feature set scores {lr_acc:.3f} "
                  f"accuracy, which is identical to the majority baseline of {base_acc:.3f}. "
                  "On accuracy alone you would throw the model away. Its recall on failing "
                  f"students is {lr_rec:.0%} against the baseline's zero, so it catches "
                  "roughly seven in ten at-risk students while the baseline catches none. "
                  "The two models score the same and are not remotely equally useful.",
                  f"""
<div class="compare">
  <div class="cmp">
    <div class="cmp-h">Majority baseline</div>
    <div class="cmp-row"><span>Accuracy</span><b>{base_acc:.3f}</b></div>
    <div class="cmp-row"><span>Failing students caught</span><b class="zero">0%</b></div>
    <div class="cmp-foot">Predicts pass for everyone. Useless, but looks respectable.</div>
  </div>
  <div class="cmp accent">
    <div class="cmp-h">Logistic regression, early features</div>
    <div class="cmp-row"><span>Accuracy</span><b>{lr_acc:.3f}</b></div>
    <div class="cmp-row"><span>Failing students caught</span><b class="good">{lr_rec:.0%}</b></div>
    <div class="cmp-foot">Same accuracy. Entirely different value.</div>
  </div>
</div>""")

    css = """
* { box-sizing:border-box; }
body { margin:0; background:#F7F8F9; color:#1F2933;
  font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; margin:0 auto; padding:52px 26px 90px; }
h1 { font-size:31px; margin:0 0 10px; font-weight:600; letter-spacing:-.5px; }
.sub { color:#7B8794; font-size:14.5px; line-height:1.65; max-width:780px; margin-bottom:26px; }
.tabs { display:flex; gap:8px; margin-bottom:30px; border-bottom:1px solid #E4E7EB; }
.tab { appearance:none; border:0; background:none; cursor:pointer; padding:12px 20px;
  font-size:14px; font-family:inherit; color:#7B8794; border-bottom:2px solid transparent;
  margin-bottom:-1px; transition:color .15s, border-color .15s; }
.tab:hover { color:#1F2933; }
.tab.on { color:#C2410C; border-bottom-color:#C2410C; font-weight:600; }
.page { display:none; } .page.on { display:block; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:16px; margin-bottom:22px; }
.card { background:#fff; border:1px solid #E4E7EB; border-radius:14px; padding:20px 22px;
  transition:box-shadow .2s ease; }
.card:hover { box-shadow:0 2px 14px rgba(31,41,51,.06); }
.card .k { font-size:11px; color:#9AA5B1; text-transform:uppercase; letter-spacing:.9px;
  font-weight:600; }
.card .v { font-size:34px; font-weight:300; margin:8px 0 3px; letter-spacing:-1px; }
.card .s { font-size:12.5px; color:#9AA5B1; }
.legendbar { display:flex; gap:26px; align-items:center; flex-wrap:wrap; font-size:12.5px;
  color:#7B8794; margin:0 0 12px 4px; }
.sw { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:7px;
  vertical-align:-1px; }
.sec { display:flex; gap:22px; }
.rail { flex:0 0 40px; display:flex; flex-direction:column; align-items:center; padding-top:26px; }
.num { width:34px; height:34px; border-radius:50%; border:1px solid #E4E7EB; background:#fff;
  color:#7B8794; display:flex; align-items:center; justify-content:center; font-size:14px;
  font-weight:600; flex:0 0 auto; }
.rail:after { content:""; flex:1; width:1px; border-left:1px dashed #D8DDE3; margin:10px 0; }
.sec:last-child .rail:after { display:none; }
.secbody { flex:1; background:#fff; border:1px solid #E4E7EB; border-radius:16px;
  padding:26px 28px; margin-bottom:18px; display:grid; grid-template-columns:300px 1fr;
  gap:30px; align-items:start; }
.sectext h3 { margin:0 0 10px; font-size:17px; font-weight:600; letter-spacing:-.2px;
  line-height:1.35; }
.sectext p { margin:0; font-size:13.5px; line-height:1.7; color:#52606D; }
.secvis { min-width:0; }
.twoup { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.timeline { padding:4px 0; }
.tl-row { display:grid; grid-template-columns:170px 1fr; gap:16px; align-items:center;
  padding:13px 0; border-bottom:1px dashed #E4E7EB; }
.tl-row:last-of-type { border-bottom:0; }
.tl-label { font-size:12px; color:#9AA5B1; text-transform:uppercase; letter-spacing:.7px;
  font-weight:600; }
.tl-items span { display:inline-block; font-size:12.5px; padding:5px 11px; border-radius:20px;
  margin:3px 6px 3px 0; }
.tl-items .ok { background:#F0F2F4; color:#52606D; }
.tl-items .bad { background:#FBEAE3; color:#C2410C; }
.tl-note { font-size:12.5px; color:#7B8794; margin-top:14px; line-height:1.6; }
.compare { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.cmp { border:1px solid #E4E7EB; border-radius:14px; padding:20px 22px; background:#FAFBFC; }
.cmp.accent { border-color:#F0D5C8; background:#FEFAF8; }
.cmp-h { font-size:13px; font-weight:600; margin-bottom:14px; }
.cmp-row { display:flex; justify-content:space-between; align-items:baseline;
  padding:9px 0; border-bottom:1px dashed #E4E7EB; font-size:13px; color:#52606D; }
.cmp-row b { font-size:20px; font-weight:400; color:#1F2933; }
.cmp-row b.zero { color:#9AA5B1; } .cmp-row b.good { color:#C2410C; }
.cmp-foot { font-size:12.5px; color:#9AA5B1; margin-top:13px; line-height:1.55; }
.note { background:#fff; border:1px solid #E4E7EB; border-left:3px solid #C2410C;
  padding:18px 22px; margin-top:24px; font-size:13.5px; line-height:1.7; color:#52606D;
  border-radius:14px; }
.note b { color:#1F2933; }
@media (max-width:900px) {
  .secbody { grid-template-columns:1fr; gap:18px; }
  .twoup, .compare { grid-template-columns:1fr; }
}
/* Mobile: strip the nested boxes back to a single flat column. Stacked
   rounded cards inside rounded sections inside a padded page reads as clutter
   on a narrow screen, so below 640px the section borders come off and the
   numbered rail moves inline with the heading. */
@media (max-width:640px) {
  .wrap { padding:26px 13px 56px; }
  h1 { font-size:23px; letter-spacing:-.3px; }
  .sub { font-size:13.5px; margin-bottom:20px; }
  .tabs { gap:2px; }
  .tab { padding:11px 13px; font-size:13.5px; flex:1; }
  .kpis { grid-template-columns:1fr 1fr; gap:9px; margin-bottom:16px; }
  .card { padding:13px 14px; border-radius:11px; }
  .card .k { font-size:9.5px; letter-spacing:.5px; }
  .card .v { font-size:25px; margin:5px 0 2px; letter-spacing:-.5px; }
  .card .s { font-size:11px; }
  .legendbar { gap:10px 16px; font-size:11.5px; margin-left:2px; }
  .sec { display:block; }
  .rail { display:none; }
  .secbody { display:block; border:0; border-radius:0; background:none;
             padding:0; margin-bottom:30px; }
  .sectext { margin-bottom:14px; }
  .sectext h3 { font-size:16px; }
  .sectext h3:before { content:counter(sec) ". "; color:#C2410C; font-weight:600; }
  .page { counter-reset:sec; }
  .sec { counter-increment:sec; }
  .sectext p { font-size:13px; line-height:1.65; }
  /* charts get their own card so they still read as discrete objects */
  .secvis { background:#fff; border:1px solid #E4E7EB; border-radius:12px;
            padding:10px 4px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
  .twoup { gap:12px; }
  .twoup > div { border-top:1px dashed #E4E7EB; padding-top:8px; }
  .twoup > div:first-child { border-top:0; padding-top:0; }
  .timeline, .compare { background:#fff; border:1px solid #E4E7EB;
                        border-radius:12px; padding:14px 15px; }
  .secvis:has(.timeline), .secvis:has(.compare) { background:none; border:0;
                                                  padding:0; }
  .tl-row { grid-template-columns:1fr; gap:6px; padding:11px 0; }
  .tl-label { font-size:10.5px; }
  .tl-items span { font-size:12px; padding:4px 10px; }
  .cmp { padding:15px 16px; }
  .cmp-row b { font-size:18px; }
  .note { padding:15px 16px; font-size:12.8px; border-radius:12px; }
}
"""

    js = """
function resizePlots(){
  if (window.Plotly) {
    document.querySelectorAll('.plotly-graph-div').forEach(function(d){
      try { Plotly.Plots.resize(d); } catch(e) {}
    });
  }
}
window.addEventListener('resize', resizePlots);
window.addEventListener('orientationchange', function(){ setTimeout(resizePlots, 200); });

document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click', function(){
    document.querySelectorAll('.tab').forEach(function(x){ x.classList.remove('on'); });
    document.querySelectorAll('.page').forEach(function(x){ x.classList.remove('on'); });
    t.classList.add('on');
    document.getElementById(t.dataset.page).classList.add('on');
    resizePlots();
    window.scrollTo({top:0, behavior:'smooth'});
  });
});
"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Performance: Risk Factors and Target Leakage</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{css}</style></head><body><div class="wrap">
<h1>Student Performance</h1>
<div class="sub">Which characteristics are associated with exam failure, and how much of a
model's apparent accuracy disappears once the target stops leaking into the features.
Data: UCI Student Performance (Cortez and Silva, 2008), adapted. See the repository README.</div>

<div class="tabs">
  <button class="tab on" data-page="p1">Risk factors</button>
  <button class="tab" data-page="p2">The leakage story</button>
</div>

<div class="page on" id="p1">{p1}
<div class="note"><b>A note on method.</b> These panels chart fail rate by category rather
than average score split by pass or fail. The pass or fail flag is derived from the subject
grades, so the second of those restates arithmetic instead of showing anything. Category
bands are mapped from the original numeric codes so they sort in their natural order, and
sample sizes appear in every label because several categories are small enough that their
rates should not be trusted.</div>
</div>

<div class="page" id="p2">{p2}
<div class="note"><b>Why this matters beyond this dataset.</b> Leakage is a timing problem,
not a column problem. A feature is legitimate only if its value would genuinely be known at
the moment the prediction has to be made. The same failure mode appears in finance when a
model is fed data from inside the very window it is predicting, in medicine when a treatment
code implies the diagnosis, and anywhere a scaler is fitted on the full dataset before
splitting. The tell is always the same: results that look too good, and a feature that could
not have been observed in time.</div>
</div>

</div><script>{js}</script></body></html>"""


def main():
    df = load()
    OUT_PATH.write_text(build_html(df), encoding="utf-8")
    base = df.is_fail.mean()
    print(f"Wrote {OUT_PATH.resolve()}")
    print(f"  {len(df)} students | overall fail rate {base*100:.1f}%")
    print("  Two pages: Risk factors, The leakage story")


if __name__ == "__main__":
    main()
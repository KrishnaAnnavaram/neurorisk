"""Turn nested-CV results into a JSON record, a Markdown report and a model card."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SYNTHETIC_DATA_NOTICE, __version__, explain, metrics
from .config import Settings
from .data import Dataset
from .features import FeatureSet
from .nested_cv import NestedCVResult
from .risk import RiskBands


def summarize_result(result: NestedCVResult, settings: Settings) -> dict:
    y, p = result.y, result.oof_probability
    out = {
        "feature_set": result.feature_set,
        "model": result.model,
        "metrics": metrics.summarize(y, p, settings.n_bootstrap, settings.seed),
        "calibration_table": metrics.calibration_table(y, p),
        "decision_curve": metrics.decision_curve(y, p),
        "risk_bands": RiskBands(settings.risk_bands).table(y, p),
        "folds": [{"fold": f.fold, "params": {k: _jsonable(v) for k, v in f.params.items()},
                   "inner_auroc": f.inner_auc, "test_rows": int(len(f.test_index))} for f in result.folds],
    }
    if result.folds and result.folds[0].domain_importance is not None:
        out["domain_importance"] = explain.average_importance([f.domain_importance for f in result.folds])
        out["feature_importance"] = explain.average_importance([f.feature_importance for f in result.folds])
    return out


def compare(results: dict[str, NestedCVResult], settings: Settings, reference: str = "A") -> list[dict]:
    """Paired bootstrap AUROC differences against the reference set, Holm-adjusted."""
    if reference not in results:
        return []
    base = results[reference]
    diffs = []
    for name, res in results.items():
        if name == reference:
            continue
        if not np.array_equal(res.y, base.y):
            raise ValueError("paired comparison needs the same rows in the same order")
        d = metrics.paired_bootstrap(base.y, base.oof_probability, res.oof_probability, "auroc",
                                     settings.n_bootstrap, settings.seed)
        diffs.append({"comparison": f"{name} - {reference}", **d.to_dict()})
    adjusted = metrics.holm([d["p_value"] for d in diffs]) if diffs else []
    for d, adj in zip(diffs, adjusted):
        d["p_adjusted"] = adj
    return diffs


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _fmt(est: dict) -> str:
    return f"{est['value']:.3f} [{est['low']:.3f}, {est['high']:.3f}]"


def render_markdown(summaries: list[dict], comparisons: list[dict], dataset: Dataset, settings: Settings,
                    feature_sets: dict[str, FeatureSet]) -> str:
    lines = [
        "# neurorisk evaluation report",
        "",
        f"> **Synthetic data.** {SYNTHETIC_DATA_NOTICE}",
        "",
        f"- Data source: `{dataset.source}` ({len(dataset.frame)} rows, prevalence {dataset.y.mean():.3f}, "
        f"sha256 `{dataset.sha256[:12]}`)",
        f"- Nested CV: {settings.outer_folds} outer folds (grouped by patient), {settings.inner_folds} inner folds, "
        f"seed {settings.seed}, calibration `{settings.calibration}`, tuning `{settings.tuning_backend}`",
        f"- Confidence intervals: 95% stratified bootstrap of out-of-fold predictions ({settings.n_bootstrap} resamples)",
        f"- neurorisk {__version__}",
        "",
        "## Discrimination and calibration (out-of-fold)",
        "",
        "| Feature set | Role | Model | AUROC | AUPRC | Brier | ECE | Cal. slope |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        m = s["metrics"]
        role = feature_sets[s["feature_set"]].role
        lines.append(f"| {s['feature_set']} | {role} | {s['model']} | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | "
                     f"{_fmt(m['brier'])} | {_fmt(m['ece'])} | {m['calibration_slope']:.2f} |")
    if comparisons:
        lines += ["", "## Paired AUROC differences (Holm-adjusted)", "",
                  "| Comparison | Difference | 95% CI | p | p (Holm) |", "|---|---|---|---|---|"]
        for c in comparisons:
            lines.append(f"| {c['comparison']} | {c['difference']:+.3f} | [{c['low']:+.3f}, {c['high']:+.3f}] | "
                         f"{c['p_value']:.4f} | {c['p_adjusted']:.4f} |")
    for s in summaries:
        lines += ["", f"## Risk bands, set {s['feature_set']} / {s['model']}", "",
                  "| Band | Probability range | Rows | Mean predicted | Observed rate |", "|---|---|---|---|---|"]
        for b in s["risk_bands"]:
            mp = "-" if b["mean_predicted"] is None else f"{b['mean_predicted']:.3f}"
            ob = "-" if b["observed_rate"] is None else f"{b['observed_rate']:.3f}"
            lines.append(f"| {b['band']} | {b['low']:.2f}-{b['high']:.2f} | {b['count']} | {mp} | {ob} |")
        if "domain_importance" in s:
            lines += ["", "| Domain | Mean AUROC drop (held-out folds) | Fold SD |", "|---|---|---|"]
            for r in s["domain_importance"]:
                lines.append(f"| {r['name']} | {r['mean_auroc_drop']:.4f} | {r['fold_std']:.4f} |")
    lines += ["", "## How to read this report", "",
              "- Set A answers the research question: do cardiometabolic factors alone predict the label?",
              "- Set C contains cognitive tests and symptoms of the disease. Its score is a ceiling, not a risk-factor result.",
              "- Risk bands are labels on a calibrated probability. The observed rate per band is the evidence for the label.",
              ""]
    return "\n".join(lines)


def render_model_card(summary: dict, dataset: Dataset, settings: Settings, fs: FeatureSet) -> str:
    m = summary["metrics"]
    return "\n".join([
        f"# Model card: neurorisk set {fs.name} / {summary['model']}",
        "",
        "## Intended use",
        "",
        "Research and teaching on tabular risk modelling. It is not a medical device, not a diagnostic tool and",
        "not a screening tool. A qualified clinician must review every individual decision.",
        "",
        "## Data",
        "",
        f"- Source: `{dataset.source}`, {len(dataset.frame)} rows, label prevalence {dataset.y.mean():.3f}.",
        f"- {SYNTHETIC_DATA_NOTICE}",
        "- Real cohorts differ in prevalence, measurement and population. Probabilities do not transfer without recalibration.",
        "",
        "## Inputs",
        "",
        f"Feature set {fs.name} ({fs.role}): {fs.description}",
        "",
        *[f"- {domain}: {', '.join(cols)}" for domain, cols in fs.domains.items()],
        "",
        "## Performance (nested CV, out-of-fold, 95% bootstrap CI)",
        "",
        f"- AUROC {_fmt(m['auroc'])}, AUPRC {_fmt(m['auprc'])}",
        f"- Brier {_fmt(m['brier'])}, ECE {_fmt(m['ece'])}, calibration slope {m['calibration_slope']:.2f}",
        "",
        "## Limits and bias",
        "",
        "- The training data is synthetic. No clinical or biological conclusion follows from these numbers.",
        "- Subgroup performance (sex, ethnicity, age) is not validated.",
        "- Permutation importance shows what the model uses, not what causes the disease.",
        "",
    ])


def write_outputs(out_dir: Path, summaries: list[dict], comparisons: list[dict], dataset: Dataset,
                  settings: Settings, feature_sets: dict[str, FeatureSet]) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": out_dir / "results.json",
        "report": out_dir / "report.md",
    }
    record = {"neurorisk_version": __version__, "data_sha256": dataset.sha256, "data_source": dataset.source,
              "rows": len(dataset.frame), "settings": {k: str(v) for k, v in settings.__dict__.items()},
              "notice": SYNTHETIC_DATA_NOTICE, "results": summaries, "comparisons": comparisons}
    paths["results"].write_text(json.dumps(record, indent=2, default=_jsonable), encoding="utf-8")
    paths["report"].write_text(render_markdown(summaries, comparisons, dataset, settings, feature_sets),
                               encoding="utf-8")
    for s in summaries:
        p = out_dir / f"model_card_{s['feature_set']}_{s['model']}.md"
        p.write_text(render_model_card(s, dataset, settings, feature_sets[s["feature_set"]]), encoding="utf-8")
        paths[f"model_card_{s['feature_set']}"] = p
    return paths

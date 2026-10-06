"""Command line interface: `neurorisk <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import SYNTHETIC_DATA_NOTICE, __version__, report, synthetic
from .config import ConfigError, Settings, parse_bands
from .data import load, source_for
from .features import load_feature_sets
from .models import MODELS
from .nested_cv import run_nested_cv
from .schema import SchemaError, validate
from .stats import compare_groups
from .training import load_model, save, train_final


def _settings(args) -> Settings:
    base = Settings.from_env()
    return base.with_overrides(
        seed=getattr(args, "seed", None),
        outer_folds=getattr(args, "outer_folds", None),
        inner_folds=getattr(args, "inner_folds", None),
        n_bootstrap=getattr(args, "bootstrap", None),
        calibration=getattr(args, "calibration", None),
        tuning_backend=getattr(args, "tuning", None),
        risk_bands=parse_bands(args.bands) if getattr(args, "bands", None) else None,
    )


def _dataset(args, settings: Settings):
    path = args.data or (str(settings.data_path) if settings.data_path else None)
    if path is None or path == "synthetic":
        print(f"[neurorisk] no --data given: using {args.synthetic_rows} synthetic rows (seed {settings.seed}).",
              file=sys.stderr)
    dataset = load(source_for(path, args.synthetic_rows, settings.seed))
    for w in dataset.report.warnings:
        print(f"[neurorisk] warning: {w}", file=sys.stderr)
    return dataset


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="CSV path, or 'synthetic' (default: NEURORISK_DATA, else synthetic)")
    p.add_argument("--synthetic-rows", type=int, default=2000, help="rows for the synthetic table")
    p.add_argument("--seed", type=int)
    p.add_argument("--outer-folds", type=int)
    p.add_argument("--inner-folds", type=int)
    p.add_argument("--bootstrap", type=int, help="bootstrap resamples for confidence intervals")
    p.add_argument("--calibration", choices=["sigmoid", "isotonic"])
    p.add_argument("--tuning", choices=["grid", "optuna"])
    p.add_argument("--bands", help="risk band cut-offs, for example 0.1,0.3,0.6")


def cmd_synth(args) -> int:
    df = synthetic.generate(args.rows, args.seed, args.missing_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} synthetic rows to {out} (prevalence {df['Diagnosis'].mean():.3f})")
    return 0


def cmd_validate(args) -> int:
    df = pd.read_csv(args.data)
    rep = validate(df, require_label=not args.no_label)
    print(f"rows: {rep.rows}")
    for w in rep.warnings:
        print(f"warning: {w}")
    for e in rep.errors:
        print(f"error: {e}")
    print("schema: OK" if rep.ok else "schema: FAILED")
    return 0 if rep.ok else 1


def _run(args, feature_set_names: list[str], models: list[str], explain_folds: bool):
    settings = _settings(args)
    dataset = _dataset(args, settings)
    sets = load_feature_sets(Path(args.feature_config) if args.feature_config else None)
    results, summaries = {}, []
    for name in feature_set_names:
        if name not in sets:
            raise ConfigError(f"unknown feature set {name!r}; choose from {sorted(sets)}")
        for model in models:
            print(f"[neurorisk] nested CV: set {name}, model {model} ...", file=sys.stderr)
            res = run_nested_cv(dataset, sets[name], model, settings, explain_folds=explain_folds)
            results[(name, model)] = res
            summaries.append(report.summarize_result(res, settings))
    return settings, dataset, sets, results, summaries


def _print_table(summaries: list[dict]) -> None:
    print(f"{'set':<4}{'model':<8}{'AUROC [95% CI]':<26}{'Brier':<8}{'ECE':<8}{'slope':<6}")
    for s in summaries:
        m = s["metrics"]
        auc = f"{m['auroc']['value']:.3f} [{m['auroc']['low']:.3f}, {m['auroc']['high']:.3f}]"
        print(f"{s['feature_set']:<4}{s['model']:<8}{auc:<26}{m['brier']['value']:<8.3f}"
              f"{m['ece']['value']:<8.3f}{m['calibration_slope']:<6.2f}")


def cmd_evaluate(args) -> int:
    settings, dataset, sets, results, summaries = _run(args, [args.feature_set], [args.model], args.explain)
    _print_table(summaries)
    if args.out:
        paths = report.write_outputs(Path(args.out), summaries, [], dataset, settings, sets)
        print(f"wrote {paths['report']}")
    return 0


def cmd_compare(args) -> int:
    names = [n.strip() for n in args.feature_sets.split(",") if n.strip()]
    settings, dataset, sets, results, summaries = _run(args, names, [args.model], args.explain)
    by_set = {name: res for (name, _), res in results.items()}
    comparisons = report.compare(by_set, settings, reference=names[0])
    _print_table(summaries)
    for c in comparisons:
        print(f"{c['comparison']}: AUROC diff {c['difference']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}], "
              f"p_holm={c['p_adjusted']:.4f}")
    out = Path(args.out) if args.out else settings.output_dir
    paths = report.write_outputs(out, summaries, comparisons, dataset, settings, sets)
    print(f"wrote {paths['report']} and {paths['results']}")
    print(SYNTHETIC_DATA_NOTICE)
    return 0


def cmd_train(args) -> int:
    settings = _settings(args)
    dataset = _dataset(args, settings)
    fs = load_feature_sets(Path(args.feature_config) if args.feature_config else None)[args.feature_set]
    trained = train_final(dataset, fs, args.model, settings)
    path = save(trained, Path(args.out))
    print(f"saved {path} (set {fs.name}, model {args.model}, params {trained.params})")
    print("This model has no held-out estimate. Use `neurorisk compare` for the performance numbers.")
    return 0


def cmd_predict(args) -> int:
    trained = load_model(Path(args.model))
    df = pd.read_csv(args.data)
    out = trained.predict(df)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out)} predictions to {args.out}")
    else:
        print(out.to_string(index=False, max_rows=20))
    return 0


def cmd_stats(args) -> int:
    settings = _settings(args)
    dataset = _dataset(args, settings)
    table = compare_groups(dataset.frame)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("Descriptive only: raw features, Mann-Whitney U, Holm and Benjamini-Hochberg adjustment.")
    return 0


def cmd_demo(args) -> int:
    args.data = "synthetic"
    args.feature_sets = "A,B,C"
    args.model = "logreg"
    args.explain = True
    args.bootstrap = args.bootstrap or 300
    return cmd_compare(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurorisk", description="Leakage-safe, calibrated Alzheimer's risk models.")
    parser.add_argument("--version", action="version", version=f"neurorisk {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic CSV with the dataset schema")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing-rate", type=float, default=0.0)
    p.add_argument("--out", default="data/synthetic.csv")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("validate", help="check a CSV against the column contract")
    p.add_argument("--data", required=True)
    p.add_argument("--no-label", action="store_true", help="the file has no Diagnosis column")
    p.set_defaults(func=cmd_validate)

    for name, func, helptext in (("evaluate", cmd_evaluate, "nested CV for one feature set and one model"),
                                 ("compare", cmd_compare, "nested CV for several feature sets, paired tests")):
        p = sub.add_parser(name, help=helptext)
        _add_common(p)
        if name == "evaluate":
            p.add_argument("--feature-set", default="A")
        else:
            p.add_argument("--feature-sets", default="A,B,C", help="comma list, the first one is the reference")
        p.add_argument("--model", default="logreg", choices=sorted(MODELS))
        p.add_argument("--feature-config", help="JSON file with feature sets (default: the bundled file)")
        p.add_argument("--explain", action="store_true", help="permutation importance on held-out folds")
        p.add_argument("--out", help="output folder for report.md, results.json and model cards")
        p.set_defaults(func=func)

    p = sub.add_parser("train", help="fit and save one final calibrated model on all rows")
    _add_common(p)
    p.add_argument("--feature-set", default="A")
    p.add_argument("--model", default="logreg", choices=sorted(MODELS))
    p.add_argument("--feature-config")
    p.add_argument("--out", default="models/neurorisk.joblib")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="score a CSV with a saved model")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("stats", help="descriptive case/control comparisons of raw features")
    _add_common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("demo", help="offline demo: synthetic data, sets A/B/C, logreg, report")
    _add_common(p)
    p.add_argument("--feature-config")
    p.add_argument("--out")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, SchemaError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

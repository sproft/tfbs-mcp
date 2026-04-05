#!/usr/bin/env python3
"""
Generate sweep YAML files from a template and model class __init__ defaults.

Usage:
  python scripts/cli/sweeps/create_sweeps.py \
    --template scripts/cli/sweeps/template.yaml \
    --models classes/nn/models.py \
    --out scripts/cli/sweeps/generated

The script inspects model classes in the given models.py file (skipping BaseModel),
extracts default values from each class __init__ and emits one sweep_{ClassName}.yaml
in the output directory by inserting a `parameters:` block into the provided template.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import argparse
from typing import Any


def const_value(node: ast.AST) -> Any:
    # Extract simple constant values from AST nodes. Extend as needed.
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [const_value(el) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(const_value(el) for el in node.elts)
    if hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # negative numbers: -3
        val = const_value(node.operand)
        if isinstance(val, (int, float)):
            return -val
    # fallback: for names like None/True/False in older ASTs
    if isinstance(node, ast.Name):
        if node.id == 'None':
            return None
        if node.id == 'True':
            return True
        if node.id == 'False':
            return False
    return None


def yaml_safe_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in [":", "{", "}", "[", "]", ",", "#", "&", "*", "?", "|", ">", "%", "@", "`", "\n"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def yaml_block_for_param(name: str, value: Any) -> str:
    # heuristics for mapping python default -> wandb sweep parameter block

    # Special case for 'dropout' parameters: they must sweep between 0.0 and 1.0.
    if "dropout" in name.lower() and isinstance(value, (float, int)) and not isinstance(value, bool):
        # Use uniform distribution for dropout rates, as they are typically floating point
        return f"  {name}:\n    distribution: uniform\n    min: 0.0\n    max: 1.0"

    if isinstance(value, (list, tuple)):
        # represent complex lists as a single categorical JSON-like value
        v = json.dumps(value)
        return f"  {name}:\n    distribution: categorical\n    values:\n      - {v}"
    if isinstance(value, bool):
        return f"  {name}:\n    distribution: categorical\n    values:\n      - true\n      - false"
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            mi = 0
            ma = 1
        else:
            mi = max(1, int(max(1, value // 2)))
            ma = int(max(value, value * 2))
        return f"  {name}:\n    distribution: int_uniform\n    min: {mi}\n    max: {ma}"
    if isinstance(value, float):
        mi = max(1e-6, value / 4)
        ma = max(value, value * 4)
        return f"  {name}:\n    distribution: uniform\n    min: {mi}\n    max: {ma}"
    if isinstance(value, str):
        return f"  {name}:\n    distribution: categorical\n    values:\n      - {yaml_safe_scalar(value)}"
    # fallback: single categorical scalar
    return f"  {name}:\n    distribution: categorical\n    values:\n      - {yaml_safe_scalar(value)}"


def split_template(tpl_lines: list[str]) -> tuple[str, str]:
    # Return (header, footer) where header is everything up to `parameters:`
    # and footer is the remainder after the parameters block (starting at e.g. early_terminate or command)
    if not any(line.strip().startswith("parameters:") for line in tpl_lines):
        raise SystemExit("Template does not contain 'parameters:' header")
    idx = next(i for i, l in enumerate(tpl_lines) if l.strip().startswith("parameters:"))
    header = "\n".join(tpl_lines[:idx])
    # find footer start: look for first top-level key after parameters: (early_terminate or command)
    footer_idx = next((i for i, l in enumerate(tpl_lines[idx + 1 :], start=idx + 1)
                       if l.strip().startswith("early_terminate:") or l.strip().startswith("command:")), None)
    if footer_idx is None:
        footer = ""
    else:
        footer = "\n".join(tpl_lines[footer_idx:])
    return header, footer


def collect_model_classes(models_source: str) -> list[tuple[str, dict]]:
    tree = ast.parse(models_source)
    classes: list[tuple[str, dict]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "BaseModel":
                continue
            init = None
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and n.name == "__init__":
                    init = n
                    break
            if not init:
                continue
            args = init.args
            argnames = [a.arg for a in args.args]
            defaults = args.defaults or []
            n_nondefault = len(argnames) - len(defaults)
            params = {}
            for i, name in enumerate(argnames):
                if name == 'self':
                    continue
                j = i - n_nondefault
                if j >= 0:
                    val = const_value(defaults[j])
                    params[name] = val
            classes.append((node.name, params))
    return classes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate sweep YAMLs from models.py and a template")
    p.add_argument("--template", default="scripts/cli/sweeps/template.yaml", help="Path to sweep template YAML")
    p.add_argument("--models", default="classes/nn/models.py", help="Path to models.py to inspect")
    p.add_argument("--out", default="scripts/cli/sweeps/generated", help="Output directory for generated sweep YAMLs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--datasets", default="core_median_24_rc,flank_median_24_rc,all_mean_nO_24_rc",
                   help=("Comma-separated list of dataset names to create sweeps for. "
                         "If omitted, a single sweep per model class is created. "
                         "Example: core_median_24_rc,flank_median_24_rc,all_mean_nO_24_rc"))
    p.add_argument("--include-models", default=None,
             help=("Comma-separated list of model class names to include. "
                   "If omitted, all model classes are considered."))
    p.add_argument("--exclude-models", default=None,
             help=("Comma-separated list of model class names to exclude. "
                   "Applied after include-models."))
    args = p.parse_args(argv)

    # Resolve paths. If a relative path is provided, prefer the current working
    # directory first, then fall back to the repository root (three levels up
    # from this script: scripts/cli/sweeps -> repo root).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    def resolve_path(p: str) -> str:
        if os.path.isabs(p):
            return p
        # try current working directory
        p_cwd = os.path.abspath(p)
        if os.path.exists(p_cwd):
            return p_cwd
        # try repository root
        p_repo = os.path.abspath(os.path.join(repo_root, p))
        if os.path.exists(p_repo):
            return p_repo
        # fallback: return path interpreted relative to cwd (will error later)
        return p_cwd

    tpl_path = resolve_path(args.template)
    models_path = resolve_path(args.models)
    # out directory: if relative, assume relative to repo root (so defaults work
    # when running from different CWDs)
    out_dir = os.path.abspath(args.out) if os.path.isabs(args.out) else os.path.abspath(os.path.join(repo_root, args.out))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(tpl_path):
        print(f"Template not found: {tpl_path}", file=sys.stderr)
        return 2
    with open(tpl_path, 'r') as f:
        tpl_lines = f.read().splitlines()

    header, footer = split_template(tpl_lines)

    if not os.path.exists(models_path):
        print(f"Models file not found: {models_path}", file=sys.stderr)
        return 2
    with open(models_path, 'r') as f:
        source = f.read()

    classes = collect_model_classes(source)
    if not classes:
        print("No model classes with __init__ defaults found in", models_path, file=sys.stderr)
        return 2

    # Filter classes by include/exclude args if provided
    include_arg = args.include_models
    exclude_arg = args.exclude_models
    include_set = {s.strip() for s in include_arg.split(',')} if include_arg else None
    exclude_set = {s.strip() for s in exclude_arg.split(',')} if exclude_arg else set()

    available = [name for name, _ in classes]
    if include_set is not None:
        # warn about unknown included names
        unknown = include_set - set(available)
        if unknown:
            print(f"Warning: include-models specified unknown classes: {', '.join(sorted(unknown))}", file=sys.stderr)
        classes = [(n, p) for (n, p) in classes if n in include_set]

    if exclude_set:
        classes = [(n, p) for (n, p) in classes if n not in exclude_set]

    if not classes:
        print("No model classes left after include/exclude filtering.", file=sys.stderr)
        return 2

    # parse datasets list: if provided, create a sweep per dataset per model
    datasets_arg = args.datasets
    if datasets_arg:
        datasets = [d.strip() for d in datasets_arg.split(',') if d.strip()]
    else:
        datasets = []

    for cls_name, params in classes:
        # If datasets provided, create per-dataset files; otherwise create one file per class
        target_datasets = datasets or [None]
        for ds in target_datasets:
            lines = ["parameters:"]

            for pname, pval in params.items():
                # Exclude 'input_channels' from sweep generation
                if pname == "input_channels":
                    continue
                if pname in ("kwargs", "*args"):
                    continue
                param_name = f"model.{pname}"
                try:
                    block = yaml_block_for_param(param_name, pval)
                except Exception:
                    block = f"  {param_name}:\n    distribution: categorical\n    values:\n      - {yaml_safe_scalar(pval)}"
                lines.append(block)

            header_text = header.rstrip('\n')
            params_text = "\n".join(lines)
            footer_text = footer.lstrip('\n')
            # Fill template placeholders in the footer (command block) so the
            # generated sweep's command uses the concrete model and dataset.
            footer_filled = footer_text.replace('${model}', cls_name)
            if ds is not None:
                footer_filled = footer_filled.replace('${data}', ds)
            content = header_text + "\n" + params_text + "\n\n" + footer_filled + "\n"
            # Final, global substitution pass to catch any remaining placeholders
            content = content.replace('${model}', cls_name)
            if ds is not None:
                content = content.replace('${data}', ds)

            if ds is None:
                out_name = f"sweep_{cls_name}.yaml"
            else:
                # sanitize dataset for filename
                safe_ds = ds.replace('/', '_')
                out_name = f"sweep_{cls_name}_{safe_ds}.yaml"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, 'w') as out:
                out.write(content)
            if not args.quiet:
                print("Wrote", out_path)

    if not args.quiet:
        print("Generated sweep YAMLs in", out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

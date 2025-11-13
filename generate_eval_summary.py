# print_eval_pipeline_summary.py
import re
from pathlib import Path
import json
import sys

EVAL_PATH = Path("./eval_new.py")  # update if different

if not EVAL_PATH.exists():
    print(f"Error: {EVAL_PATH} not found. Place this script next to eval_new.py or update EVAL_PATH.")
    sys.exit(1)

text = EVAL_PATH.read_text()

def find_argparse_args(s):
    pattern = r"parser\.add_argument\((.*?)\)"
    args = re.findall(pattern, s, flags=re.S)
    parsed = []
    for a in args:
        # rough parse: get first string as flag, and look for 'help', 'default', 'action'
        flags = re.findall(r"['\"](--?[a-zA-Z0-9_\-]+)['\"]", a)
        default = re.search(r"default\s*=\s*([^\),\n]+)", a)
        action = re.search(r"action\s*=\s*['\"]([^'\"]+)['\"]", a)
        helpm = re.search(r"help\s*=\s*['\"]([^'\"]*)['\"]", a)
        parsed.append({
            "flags": flags,
            "default": default.group(1).strip() if default else None,
            "action": action.group(1) if action else None,
            "help": helpm.group(1) if helpm else None
        })
    return parsed

def find_classes(s):
    return re.findall(r"class\s+([A-Za-z0-9_]+)\s*\(", s)

def find_functions(s):
    return re.findall(r"def\s+([A-Za-z0-9_]+)\s*\(", s)

def find_xai_inits(s):
    patterns = [
        ("EnhancedFDCAAttentionVisualizer", r"EnhancedFDCAAttentionVisualizer\s*\("),
        ("EnhancedSegmentationGradCAM", r"EnhancedSegmentationGradCAM\s*\("),
        ("EnhancedFrequencyComponentAnalyzer", r"EnhancedFrequencyComponentAnalyzer\s*\("),
        ("EnhancedFrequencyDomainAnalyzer", r"EnhancedFrequencyDomainAnalyzer\s*\(")
    ]
    hits = []
    for name, pat in patterns:
        m = re.search(pat, s)
        if m:
            hits.append(name)
    return hits

def find_output_dirs(s):
    # look for mkdir or save_dir usages
    out = set()
    for m in re.finditer(r"save_dir\s*=\s*([^\),\n]+)", s):
        out.add(m.group(1).strip().strip("'\""))
    # timestamped run_dir
    if "xai_" in s and "run_dir" in s:
        out.add("outputs/xai_<timestamp>/")
    return list(out)

def find_evaluate_batch_signature(s):
    m = re.search(r"def\s+evaluate_batch\s*\((.*?)\):", s, flags=re.S)
    return m.group(1).strip() if m else None

summary = {}
summary["file"] = str(EVAL_PATH)
summary["classes"] = find_classes(text)
summary["functions"] = find_functions(text)[:50]  # show top 50
summary["argparse_args"] = find_argparse_args(text)
summary["xai_modules_initialized"] = find_xai_inits(text)
summary["output_directories"] = find_output_dirs(text)
summary["evaluate_batch_signature"] = find_evaluate_batch_signature(text)
summary["main_loop_present"] = "for batch_idx" in text or "tqdm(val_loader" in text

# Print neatly
print("\n" + "="*80)
print("SUMMARY: eval_new.py (evaluation pipeline)")
print("="*80 + "\n")

print("1) Top-level classes found:")
for c in summary["classes"]:
    print("   -", c)
print()

print("2) Notable functions (first 50):")
for f in summary["functions"][:50]:
    print("   -", f)
print()

print("3) CLI arguments (parsed from argparse.add_argument calls):")
for a in summary["argparse_args"]:
    print("   - flags:", a["flags"], "| action:", a["action"], "| default:", a["default"], "| help:", a["help"])
print()

print("4) XAI modules initialized (detected):")
for m in summary["xai_modules_initialized"]:
    print("   -", m)
print()

print("5) Output directories / save_dir patterns detected:")
for d in summary["output_directories"]:
    print("   -", d)
print()

print("6) evaluate_batch signature (inputs expected):")
print("   -", summary["evaluate_batch_signature"])
print()

print("7) Does file contain a main evaluation loop over loader?:", summary["main_loop_present"])
print()

print("8) Quick guidance (auto):")
print("   - Ensure the dataset loader returns tensors in the exact indexing eval_new expects (low channels first).")
print("   - Make sure your model has modules named with 'fdca' or 'fusion' if you rely on hooks for mechanistic traces.")
print("   - Output images will be saved under a timestamped run dir, e.g. ./outputs/xai_YYYY-MM-DD_HH-MM-SS/")
print()

print("="*80 + "\n")

# Also emit JSON file for programmatic use
out_path = Path("eval_pipeline_summary.json")
out_path.write_text(json.dumps(summary, indent=2))
print(f"Structured JSON summary saved to {out_path.resolve()}")

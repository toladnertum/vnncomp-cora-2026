#!/usr/bin/env python3
"""
Independent counterexample validator for VNN-LIB 2.0 multi-network benchmarks
(monotonic_acasxu_2026, isomorphic_acasxu_2026).

The official VNN-COMP checker does not support 2.0 list-style ONNX fields yet,
so this validates CORA's multi-network counterexamples independently:
  - parse the CE file (X_f/Y_f/X_g/Y_g blocks),
  - re-execute each sub-network with onnxruntime (authoritative outputs),
  - evaluate every (assert ...) of the 2.0 vnnlib on the CE assignment.
A counterexample is valid iff every assertion (input box, cross-network
coupling, and the output property) holds.

Usage:
    python3 validate_ces_v2.py <benchmark_dir_2.0> <ce_file> [ce_file ...]
    python3 validate_ces_v2.py <benchmark_dir_2.0> --instances   # run all from instances.csv (needs CE dir)

deps: onnx, onnxruntime, numpy
"""
import sys, os, re, gzip, ast
import numpy as np
import onnx
import onnxruntime as ort

ATOL = 1e-4   # input-box / equality tolerance
RTOL = 1e-3   # onnx-vs-CE output match tolerance


def _open(path):
    if os.path.isfile(path):
        return open(path, "r").read()
    if os.path.isfile(path + ".gz"):
        return gzip.open(path + ".gz", "rt").read()
    raise FileNotFoundError(path)


# ----- counterexample file -------------------------------------------------
def parse_ce(text):
    """Parse 'sat\\nX_f real [5]\\n<vals>\\nY_f ...' into {name: np.array}."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or lines[0].lower() != "sat":
        return None
    blocks, cur = {}, None
    hdr = re.compile(r"^([A-Za-z_]\w*)\s+\w+\s+\[[\d,]*\]$")
    for l in lines[1:]:
        m = hdr.match(l)
        if m:
            cur = m.group(1)
            blocks[cur] = []
        elif cur is not None:
            blocks[cur].append(float(l))
    return {k: np.array(v, dtype=np.float64) for k, v in blocks.items()}


# ----- onnx ----------------------------------------------------------------
def resolve_onnx(p):
    """Mirror CORA's path resolution: as-is, +.gz, or strip one intermediate
    dir (onnx/original/x -> onnx/x), decompressing .gz to a usable .onnx."""
    cands = [p, p + ".gz"]
    parts = p.split("/")
    if len(parts) >= 2:
        # drop the dir immediately containing the file: onnx/original/x -> onnx/x
        stripped = "/".join(parts[:-2] + parts[-1:])
        cands += [stripped, stripped + ".gz"]
    for c in cands:
        if c.endswith(".gz") and os.path.isfile(c):
            out = c[:-3]
            if not os.path.isfile(out):
                with gzip.open(c, "rb") as f, open(out, "wb") as o:
                    o.write(f.read())
            return out
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(p)


def run_onnx(onnx_path, x):
    model = onnx.load(resolve_onnx(onnx_path))
    sess = ort.InferenceSession(model.SerializeToString())
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    xin = x.reshape(shape).astype(np.float32)
    out = sess.run(None, {inp.name: xin})[0]
    return np.array(out, dtype=np.float64).flatten()


# ----- vnnlib 2.0 assertion evaluator --------------------------------------
def tokenize(s):
    return re.findall(r"\(|\)|[^\s()]+", s)


def parse_sexprs(tokens):
    """Return list of top-level s-expressions (nested python lists/atoms)."""
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]; pos += 1
        if tok == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(parse())
            pos += 1
            return lst
        return tok

    out = []
    while pos < len(tokens):
        out.append(parse())
    return out


def get_asserts_and_nets(vnnlib_text):
    # strip comments
    text = "\n".join(l.split(";", 1)[0] for l in vnnlib_text.splitlines())
    sexprs = parse_sexprs(tokenize(text))
    asserts, nets = [], []
    for e in sexprs:
        if isinstance(e, list) and e:
            if e[0] == "assert":
                asserts.append(e[1])
            elif e[0] == "declare-network":
                nets.append(e[1])
    return asserts, nets


def evaluate(node, env):
    """Evaluate a vnnlib assertion node to bool/float using env {var: value}."""
    if isinstance(node, str):
        if re.match(r"^[A-Za-z_]\w*\[\d+\]$", node):   # variable ref X_f[0]
            return env[node]
        return float(node)                              # numeric constant
    # list form: prefix (op a b ...) or infix (a op b)
    ops = {"<=": lambda a, b: a <= b + ATOL,
           ">=": lambda a, b: a >= b - ATOL,
           "<": lambda a, b: a < b,
           ">": lambda a, b: a > b,
           "==": lambda a, b: abs(a - b) <= ATOL,
           "+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b}
    if len(node) == 3 and isinstance(node[1], str) and node[1] in ops:  # infix: (a < b)
        return ops[node[1]](evaluate(node[0], env), evaluate(node[2], env))
    head = node[0]
    if head == "and":
        return all(evaluate(c, env) for c in node[1:])
    if head == "or":
        return any(evaluate(c, env) for c in node[1:])
    if head in ops:                                     # prefix: (<= a b)
        return ops[head](evaluate(node[1], env), evaluate(node[2], env))
    raise ValueError(f"unknown node: {node}")


def is_input_assert(node):
    """True if the assertion only references input vars (X_*)."""
    s = str(node)
    return "X_" in s and "Y_" not in s


# ----- per-instance validation ---------------------------------------------
def build_env(ce, y_by_net, net_names):
    env = {}
    for nm in net_names:
        xk, yk = f"X_{nm}", f"Y_{nm}"
        for i, v in enumerate(ce[xk]):
            env[f"{xk}[{i}]"] = v
        for i, v in enumerate(y_by_net[nm]):
            env[f"{yk}[{i}]"] = v
    return env


def validate(ce_text, vnnlib_text, onnx_paths):
    ce = parse_ce(ce_text)
    if ce is None:
        return "no_ce", "result not sat or empty CE"
    asserts, net_names = get_asserts_and_nets(vnnlib_text)
    # re-execute each sub-network on the CE input
    y_by_net, exec_msgs = {}, []
    for nm in net_names:
        y = run_onnx(onnx_paths[nm], ce[f"X_{nm}"])
        y_by_net[nm] = y
        if f"Y_{nm}" in ce:
            diff = np.max(np.abs(y - ce[f"Y_{nm}"]))
            norm = max(np.max(np.abs(ce[f"Y_{nm}"])), 1e-6)
            if diff / norm > RTOL:
                exec_msgs.append(f"{nm}: |onnx-CE|={diff:.2e}")
    if exec_msgs:
        return "exec_doesnt_match", "; ".join(exec_msgs)
    env = build_env(ce, y_by_net, net_names)
    in_ok = all(evaluate(a, env) for a in asserts if is_input_assert(a))
    out_ok = all(evaluate(a, env) for a in asserts if not is_input_assert(a))
    if not in_ok:
        return "ce_outside_input", "input/coupling assertion violated"
    if not out_ok:
        return "spec_not_violated", "output property holds (not a counterexample)"
    return "correct", "all assertions satisfied"


# ----- driver --------------------------------------------------------------
def parse_model_field(s):
    """[('f','pathF'),('g','pathG')] -> {'f':pathF,'g':pathG}."""
    pairs = re.findall(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", s)
    return {role: path for role, path in pairs}


def main():
    bench_dir = sys.argv[1]
    import csv
    inst = {}
    with open(os.path.join(bench_dir, "instances.csv")) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            nets = parse_model_field(row[0])
            vn = os.path.join(bench_dir, row[1].lstrip("./"))
            # key by the f sub-network basename (unique per instance)
            fbase = os.path.splitext(os.path.basename(nets["f"]))[0]
            inst[fbase] = (nets, vn)

    ce_files = sys.argv[2:]
    tally = {}
    for ce_file in ce_files:
        ce_text = _open(ce_file)
        # the f-network basename appears in the CE filename
        match = next((v for k, v in inst.items() if k in os.path.basename(ce_file)), None)
        if match is None and len(inst) == 1:
            match = list(inst.values())[0]
        if match is None:
            print(f"{os.path.basename(ce_file)}: NO_INSTANCE_MATCH"); continue
        nets, vn = match
        onnx_paths = {role: os.path.join(bench_dir, p.lstrip("./")) for role, p in nets.items()}
        verdict, msg = validate(ce_text, _open(vn), onnx_paths)
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"{os.path.basename(ce_file)}: {verdict} | {msg}")
    print("\n=== TALLY ===")
    for k, v in sorted(tally.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

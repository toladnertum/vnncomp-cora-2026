# VNN-LIB 2.0 multi-network counterexample validator

Independent checker for the multi-network 2.0 benchmarks (monotonic_acasxu_2026,
isomorphic_acasxu_2026). The official VNN-COMP scorer does not yet validate
2.0 list-style ONNX fields, so this verifies CORA's counterexamples standalone.

It is independent of CORA: outputs come from onnxruntime, and the vnnlib
assertions are evaluated by a small s-expression evaluator. A counterexample is
`correct` only if every assertion (input box, cross-network coupling, and the
output property) holds on the CE assignment.

## Usage

    pip install -r requirements.txt
    python3 validate_ces_v2.py <benchmark_2.0_dir> <ce_file> [<ce_file> ...]

`<benchmark_2.0_dir>` holds `instances.csv` and the `onnx/` and `vnnlib/`
directories. CE files are matched to instances by the f sub-network name.

## Verdicts

- `correct` — all assertions satisfied (valid counterexample)
- `spec_not_violated` — inputs in range but the output property holds (not a CE)
- `ce_outside_input` — an input/coupling assertion is violated
- `exec_doesnt_match` — onnxruntime output differs from the CE's written output
- `no_ce` — result not `sat` or empty CE

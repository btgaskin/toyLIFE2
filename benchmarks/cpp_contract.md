# Author C++ benchmark adapter

Build from the repository root:

```sh
mkdir -p .cache/cpp-bench
clang++ -O3 -DNDEBUG -std=c++11 -I upstream/toylife benchmarks/cpp_author.cpp upstream/toylife/toy_plugin.cpp upstream/toylife/helper_functions.cpp upstream/toylife/patterns/patternsplugin.cpp -o .cache/cpp-bench/cpp_author
```

Run with working directory `upstream/toylife`:

```text
cpp_author regulatory|metabolic|spatial_descriptor INPUT REPEATS OUTPUT [LOOPS]
```

Use absolute input/output paths. Each input line contains decimal 20-bit gene integers in gene order. Blank lines are ignored. Regulatory and metabolic inputs allow 1–16 genes; spatial descriptors require two genes. The adapter constructs fixed-width binary genotype strings before timing and calls unchanged upstream public routines.

JSON output fields are `workload`, `initialization_seconds`, `first_seconds`, `loops`, `rows`, `output`, and `durations_seconds`. Initialization measures plugin construction/table loading. First measures one complete evaluation of the input rows. Each subsequent duration measures `loops` complete evaluations. Results retain the final evaluation. Timed work includes upstream genotype parsing, phenotype computation and output-vector allocation; input parsing, table loading, JSON serialization and file I/O are excluded.

`regulatory` returns every next expression state in ascending initial-state order. `metabolic` returns four assay flags for the fixed environment `00010001`, `00000011`, `01101010`, `10100101` in that order. `spatial_descriptor` returns the author's six response states followed by the dimer-existence flag.

For independent memory/undefined-behavior checks, replace `-O3 -DNDEBUG` with `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` and use a separate output executable. Sanitizer timings are excluded from performance results.

The author source has iterator invalidation in tied-energy removal at `toy_plugin.cpp:421–426`. The adapter preserves these routines. A sanitizer failure is an execution failure and must be recorded with its input, rather than changing the source or removing that input to obtain parity.

The upstream tissue simulation is embedded in an executable that consumes precomputed cellular-automaton tables. Its neighbor access evaluates `(c-1)%cells` before boundary handling, producing a negative vector index at cell zero. This adapter therefore exposes the unchanged spatial response routine; a matched 31-cell, 64-step tissue comparison requires a separate declared implementation contract.

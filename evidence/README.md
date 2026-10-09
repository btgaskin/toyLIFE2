# Recorded evidence

These summaries preserve measurements and qualification results from the
development Mac. [The manifest](manifest.json) records original and exported
SHA-256 hashes. Checkout-absolute paths are replaced by `<checkout>`; numerical
results are preserved. Captured source identities describe the measured version.

| Summary | What it records |
| --- | --- |
| [Original C++](cpp-original.json) | Frozen inputs, source failures, and diagnostic results |
| [Repaired C++](cpp-repaired.json) | Unfiltered comparisons, differences, and timings |
| [Agreement cohorts](cpp-agreement.json) | Postselected matching outputs; ten-thread CPU and Metal timing |
| [One-thread agreement cohorts](cpp-agreement-one-thread.json) | The separate one-thread CPU comparison |
| [Gene scaling](metabolic-gene-scaling.json) | Two through 32 genes in fixed metabolic assays |
| [Evolution scaling](evolution-scaling.json) | Population requests, cache reuse, and batching controls |
| [Complete CPU map](full-map-cpu.json) | Weighted histograms and enumeration protocol |
| [CPU–Metal comparison](cpu-metal-comparison.json) | Saved map equality and backend timings |
| [CPU sessions](sessions-cpu.json) | Integrated control, population, and checkpoint checks |
| [Metal sessions](sessions-metal.json) | Physical Metal checks for the same session qualification |

The [benchmark guide](../docs/cpp-benchmark.md) defines timing boundaries,
C++ repairs, agreement selection, hardware, and reproduction commands.
[Scientific qualification](../docs/qualification.md) tracks model validation
and the outstanding differences from published aggregate counts.

The generated `qualification/` directory contains full local run records,
inputs, arrays, compiler products, and logs. It is excluded from Git. These
summaries provide the evidence cited in the README; the scripts and pinned
upstream source allow new runs. Hosted CPU CI is separate from these historical
measurements and from physical Metal qualification.

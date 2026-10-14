"""Single-event point/gene-copy proposals, independent of fitness effects."""

from __future__ import annotations

import numpy as np

from .genotypes import GenomeBatch

PROTOCOL = "point-gene-copy-single-event-v1"


def gene_copy_step(
    population, weights, *, stream, replicate, generation, mutation_probability, gene_copy
):
    from .population import wright_fisher_step

    batch = population
    if batch.genes.ndim != 2 or batch.genes.shape[-1] < gene_copy.max_genes:
        raise ValueError("gene-copy proposals require a population padded to max_genes")
    new, parents, _, _ = wright_fisher_step(
        batch.genes,
        weights,
        stream=stream,
        replicate=replicate,
        generation=generation,
        mutation_probability=0,
    )
    counts = batch.gene_counts[parents].copy()
    ids = np.arange(len(new), dtype=np.uint64)
    u = stream.uniform(ids, role="mutation_gate", trial=replicate, event=generation)
    upper = mutation_probability + gene_copy.duplication_probability
    operations = np.where(
        u < mutation_probability,
        1,
        np.where(u < upper, 2, np.where(u < upper + gene_copy.deletion_probability, 3, 0)),
    ).astype(np.int32)
    positions = np.full(len(new), -1, np.int32)
    blocked = np.zeros(len(new), bool)
    for n in np.unique(counts):
        members = np.flatnonzero(counts == n)
        points = members[operations[members] == 1]
        if len(points):
            bits = stream.integers(
                20 * int(n), ids[points], role="mutation_bit", trial=replicate, event=generation
            ).astype(np.int64)
            positions[points] = bits
            new[points, bits // 20] ^= np.uint32(1) << (19 - bits % 20).astype(np.uint32)
        for code, role in ((2, "duplication_gene"), (3, "deletion_gene")):
            selected = members[operations[members] == code]
            if not len(selected):
                continue
            targets = stream.integers(
                int(n), ids[selected], role=role, trial=replicate, event=generation
            )
            for j, target in zip(selected, targets, strict=True):
                k, length = int(target), int(n)
                positions[j] = k
                blocked[j] = (code == 2 and length == gene_copy.max_genes) or (
                    code == 3 and length == gene_copy.min_genes
                )
                if blocked[j]:
                    continue
                active = new[j, :length].copy()
                row = np.insert(active, k + 1, active[k]) if code == 2 else np.delete(active, k)
                new[j] = 0
                new[j, : len(row)] = row
                counts[j] = len(row)
    result = GenomeBatch(new, counts, min_genes=gene_copy.min_genes, max_genes=gene_copy.max_genes)
    return result, parents, dict(operations=operations, positions=positions, blocked=blocked)

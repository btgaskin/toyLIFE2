"""Ordered genes: four promoter bits followed by sixteen coding bits; 0=P, 1=H."""

from __future__ import annotations

import numpy as np


def encode(sequences: str | list[str] | tuple[str, ...]) -> np.ndarray:
    scalar = isinstance(sequences, str)
    seqs = [sequences] if scalar else list(sequences)
    if not seqs or not isinstance(seqs[0], str) or not seqs[0] or len(seqs[0]) % 20:
        raise ValueError("genotypes must contain a positive multiple of twenty bits")
    length = len(seqs[0])
    if any(len(s) != length or set(s) - {"0", "1"} for s in seqs):
        raise ValueError("genotypes must be equal-length binary strings")
    result = np.asarray(
        [[int(s[g : g + 20], 2) for g in range(0, length, 20)] for s in seqs], dtype=np.uint32
    )
    return result[0] if scalar else result


def validate(genotypes: np.ndarray) -> np.ndarray:
    raw = np.asarray(genotypes)
    if raw.ndim < 1 or raw.shape[-1] < 1 or raw.size == 0:
        raise ValueError("genotypes require nonempty leading axes and a final gene axis")
    if raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= 1 << 20):
        raise ValueError("each gene must be an integer in [0, 2**20)")
    return np.array(raw, dtype=np.uint32, order="C", copy=True)


def decode(genotypes: np.ndarray) -> str | np.ndarray:
    genes = validate(genotypes)
    flat = genes.reshape(-1, genes.shape[-1])
    strings = ["".join(f"{int(g):020b}" for g in row) for row in flat]
    if genes.ndim == 1:
        return strings[0]
    return np.asarray(strings).reshape(genes.shape[:-1])


def mutation(genotype: np.ndarray, position: int) -> np.ndarray:
    genes = validate(genotype)
    if type(position) is not int or not 0 <= position < 20 * genes.shape[-1]:
        raise ValueError("mutation position is outside genotype")
    genes[..., position // 20] ^= np.uint32(1 << (19 - position % 20))
    return genes


def neighbours(genotypes: np.ndarray) -> np.ndarray:
    genes = validate(genotypes)
    return np.stack([mutation(genes, p) for p in range(20 * genes.shape[-1])], axis=-2)


class GenomeBatch:
    """Owned padded genes with explicit active counts; padding is never a gene."""

    def __init__(self, genes, gene_counts, *, min_genes=1, max_genes=None):
        values = validate(genes)
        counts = np.asarray(gene_counts)
        width = values.shape[-1]
        maximum = width if max_genes is None else max_genes
        if (
            counts.shape != values.shape[:-1]
            or counts.dtype.kind not in "iu"
            or not 1 <= min_genes <= maximum <= 20
            or width > 20
            or np.any(counts < min_genes)
            or np.any(counts > maximum)
            or np.any(counts > width)
        ):
            raise ValueError("invalid active gene counts or gene-copy bounds")
        if np.any(values[np.arange(width) >= counts[..., None]] != 0):
            raise ValueError("inactive gene padding must be zero")
        self.genes = values
        self.gene_counts = np.array(counts, dtype=np.int32, copy=True)
        self.genes.flags.writeable = False
        self.gene_counts.flags.writeable = False

    @classmethod
    def from_fixed(cls, genes, width):
        values = validate(genes)
        padded = np.zeros((*values.shape[:-1], width), np.uint32)
        padded[..., : values.shape[-1]] = values
        return cls(padded, np.full(values.shape[:-1], values.shape[-1], np.int32))

    def __len__(self):
        return len(self.gene_counts)

    def __getitem__(self, index):
        return GenomeBatch(self.genes[index], self.gene_counts[index])

    def active(self, index):
        return self.genes[index][: int(self.gene_counts[index])].copy()

    def keys(self):
        flat = self.genes.reshape(-1, self.genes.shape[-1])
        return [
            tuple(map(int, row[: int(n)]))
            for row, n in zip(flat, self.gene_counts.flat, strict=True)
        ]

    def strings(self):
        return np.asarray(["".join(f"{gene:020b}" for gene in row) for row in self.keys()]).reshape(
            self.gene_counts.shape
        )


def load_selected(root, *, initial=False):
    """Read a versioned selection pair or an exact-width legacy selection."""
    import json
    from pathlib import Path

    root = Path(root)
    prefix = "initial_" if initial else ""
    genes = np.load(root / (prefix + "selected_genomes.npy"), allow_pickle=False)
    counts = root / (prefix + "selected_gene_counts.npy")
    manifest = root / "selection.json"
    required = manifest.exists()
    if required and json.loads(manifest.read_text()).get("format") != "gene-counts-v1":
        raise ValueError("unsupported selected-genome format")
    if required or counts.exists():
        return GenomeBatch(genes, np.load(counts, allow_pickle=False))
    return validate(genes)


def save_selected(root, selected, *, initial=False):
    import json
    from pathlib import Path

    root = Path(root)
    prefix = "initial_" if initial else ""
    np.save(
        root / (prefix + "selected_genomes.npy"),
        selected.genes if isinstance(selected, GenomeBatch) else selected,
        allow_pickle=False,
    )
    if isinstance(selected, GenomeBatch):
        np.save(
            root / (prefix + "selected_gene_counts.npy"), selected.gene_counts, allow_pickle=False
        )
        (root / "selection.json").write_text(json.dumps({"format": "gene-counts-v1"}) + "\n")

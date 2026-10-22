"""Download Oetjen et al. 2018 bone marrow scRNA-seq (GEO GSE120221) and build one counts h5ad.

25 GEO samples from 20 donors, 10x v2. Sample ids are a donor letter plus an optional replicate
suffix (C1, C2, Ck, S1, S2, Sk1, Sk2), so ``donor`` is the leading letter and ``sample`` the full id.
X holds raw integer UMI counts (CSR). Cells of donors A, P and U that the scIB immune atlas kept
get its ``final_annotation`` and ``dpt_pseudotime`` as ``scib_label`` / ``scib_dpt_pseudotime``.

Run from the repo root:
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_build_oetjen_h5ad.py"
"""

import argparse
import gzip
import re
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scipy.io

GEO_TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120221/suppl/GSE120221_RAW.tar"
GEO_SAMPLES_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi" "?acc=GSE120221&targ=gsm&form=text&view=brief"
)
HERE = Path(__file__).resolve().parent
DEFAULT_SCIB = HERE.parent / "scIB_human_immune" / "data" / "Immune_ALL_human.h5ad"


def donor_of(sample: str) -> str:
    """Donor letter of a GEO sample id: 'A' -> 'A', 'C1'/'Ck' -> 'C', 'Sk2' -> 'S'."""
    match = re.fullmatch(r"([A-Z])k?\d?", sample)
    if match is None:
        raise ValueError(f"unexpected Oetjen sample id {sample!r}")
    return match.group(1)


def parse_geo_samples(text: str) -> pd.DataFrame:
    """GEO brief sample text -> one row per GSM with sample id and donor age."""
    rows = []
    for block in text.split("^SAMPLE = ")[1:]:
        gsm = block.split("\n", 1)[0].strip()
        title = re.search(r"!Sample_title = Bone marrow (\S+)", block)
        age = re.search(r"age: (\d+) years old", block)
        if title is None or age is None:
            raise ValueError(f"{gsm}: missing title or age in GEO record")
        rows.append({"gsm": gsm, "sample": title.group(1), "age": int(age.group(1))})
    return pd.DataFrame(rows)


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp, "wb") as out:
            out.writelines(response.iter_content(chunk_size=1 << 20))
    tmp.rename(path)


def read_sample(raw_dir: Path, gsm: str, sample: str) -> ad.AnnData:
    """One 10x v2 triple (genes x cells mtx) -> cells x genes AnnData of integer counts."""
    prefix = f"{gsm}_{{}}_{sample}"
    counts = scipy.io.mmread(raw_dir / f"{prefix.format('matrix')}.mtx.gz").T.tocsr()
    genes = pd.read_csv(raw_dir / f"{prefix.format('genes')}.tsv.gz", sep="\t", header=None)
    with gzip.open(raw_dir / f"{prefix.format('barcodes')}.tsv.gz", "rt") as fh:
        barcodes = [line.strip() for line in fh]
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError(f"{sample}: matrix {counts.shape} vs {len(barcodes)} x {len(genes)}")
    var = pd.DataFrame({"gene_ids": genes[0].values}, index=genes[1].values)
    var.index = ad.utils.make_index_unique(var.index)
    return ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame(index=[f"{bc}-Oetjen_{sample}" for bc in barcodes]),
        var=var,
    )


def build(raw_dir: Path, samples: pd.DataFrame) -> ad.AnnData:
    parts = {}
    for row in samples.itertuples():
        part = read_sample(raw_dir, row.gsm, row.sample)
        part.obs["sample"] = row.sample
        part.obs["donor"] = donor_of(row.sample)
        part.obs["age"] = row.age
        part.obs["gsm"] = row.gsm
        parts[row.sample] = part
    genes = [p.var_names for p in parts.values()]
    if any(not g.equals(genes[0]) for g in genes):
        raise ValueError("samples do not share one gene list")
    adata = ad.concat(parts.values(), merge="same")
    for col in ("sample", "donor", "gsm"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.obs["n_counts"] = np.asarray(adata.X.sum(1)).ravel()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(1)).ravel()
    adata.layers["counts"] = adata.X.copy()
    return adata


def attach_scib_labels(adata: ad.AnnData, scib_path: Path) -> None:
    """Copy scIB annotation and erythroid pseudotime onto matching A/P/U barcodes (NaN elsewhere)."""
    scib = ad.read_h5ad(scib_path, backed="r").obs
    scib = scib[scib["study"] == "Oetjen"]
    matched = adata.obs_names.intersection(scib.index)
    if len(matched) != len(scib):
        raise ValueError(f"only {len(matched)} of {len(scib)} scIB Oetjen cells matched by barcode")
    adata.obs["scib_label"] = scib["final_annotation"].astype(str).reindex(adata.obs_names)
    adata.obs["scib_dpt_pseudotime"] = scib["dpt_pseudotime"].reindex(adata.obs_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--scib", type=Path, default=DEFAULT_SCIB, help="'' to skip labels")
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.data_dir / "GSE120221_RAW.tar"
    download(GEO_TAR_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir, filter="data")

    samples = parse_geo_samples(requests.get(GEO_SAMPLES_URL, timeout=60).text)
    if len(samples) != 25:
        raise ValueError(f"expected 25 GEO samples, got {len(samples)}")
    adata = build(raw_dir, samples)
    if str(args.scib):
        attach_scib_labels(adata, args.scib)

    out = args.data_dir / "oetjen_bm_counts.h5ad"
    adata.write_h5ad(out, compression="gzip")
    print(adata)
    print(
        adata.obs.groupby("sample", observed=True)
        .agg(
            donor=("donor", "first"),
            age=("age", "first"),
            cells=("n_counts", "size"),
            median_umi=("n_counts", "median"),
        )
        .to_string()
    )
    print("Saved", out)


if __name__ == "__main__":
    main()

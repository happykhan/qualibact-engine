#!/usr/bin/env python3
"""Build a merged metadata TSV from feedback datasets and run qualibact on each group.

Five groups, each treated as a single species in the qualibact pipeline:
  Achromobacter xylosoxidans   <- feedback/axylosoxidans
  Campylobacter coli           <- feedback/campy_coli
  Campylobacter jejuni         <- feedback/campy_jejuni
  Neisseria commensals         <- feedback/neiserria/N_commensals      (incl. N. meningitidis rows)
  Neisseria gonorrhoeae        <- feedback/neiserria/Ngo_for_Qualibact

Source files have inconsistent layouts — see per-loader notes below. The output schema
matches qualibact.species_run.METRICS_LIST plus species_sylph and Sample.

Refseq for "Neisseria commensals": this label is not an NCBI taxon, so the engine's
refseq fetch would return nothing (and qualibact.refseq.get_metrics will spin retrying
on a None result). Before the prepare step we pre-fetch RefSeq summaries for each
distinct species named in the commensals QUAST file and write a stitched
"Neisseria commensals.json" into the species dir; run_datasets_summary then sees the
cached file and skips the bad lookup.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qualibact.refseq import run_datasets_summary  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEEDBACK_DIR = REPO_ROOT / "feedback"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "qualibact_output_feedback"
DEFAULT_MERGED_TSV = REPO_ROOT / "feedback_metadata.tsv.gz"

REQUIRED_COLS = [
    "Sample",
    "N50",
    "number",
    "longest",
    "GC_Content",
    "Completeness_Specific",
    "Contamination",
    "Total_Coding_Sequences",
    "Genome_Size",
    "total_length",  # required by qualibact.species_util.DEF_METRIC_LIST during calculate
    "species_sylph",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_feedback")


def _from_checkm2_only(checkm2_path: Path, species: str) -> pd.DataFrame:
    """For groups whose CheckM2 file already carries Total_Contigs / Max_Contig_Length /
    Contig_N50 (axylosoxidans, Ngo, N_commensals). CheckM2 'Completeness' is mapped to
    Completeness_Specific (all rows use the Specific Model). GC is fraction -> percent."""
    df = pd.read_csv(checkm2_path, sep="\t")
    out = pd.DataFrame({
        "Sample": df["Name"].astype(str),
        "N50": df["Contig_N50"],
        "number": df["Total_Contigs"],
        "longest": df["Max_Contig_Length"],
        "GC_Content": df["GC_Content"].astype(float) * 100.0,
        "Completeness_Specific": df["Completeness"].astype(float),
        "Contamination": df["Contamination"].astype(float),
        "Total_Coding_Sequences": df["Total_Coding_Sequences"],
        "Genome_Size": df["Genome_Size"],
        "total_length": df["Genome_Size"],  # CheckM2 reports same metric under one name
        "species_sylph": species,
    })
    return out


def load_axylosoxidans(group_dir: Path) -> pd.DataFrame:
    return _from_checkm2_only(
        group_dir / "checkm2_results_ax.tsv.gz",
        "Achromobacter xylosoxidans",
    )


def load_ngo(group_dir: Path) -> pd.DataFrame:
    return _from_checkm2_only(
        group_dir / "checkM2_Ngo_LSB_TDM_MM_29825_unique.tsv",
        "Neisseria gonorrhoeae",
    )


def load_n_commensals(group_dir: Path) -> pd.DataFrame:
    # All rows kept under one label per user direction (file mixes commensals + N. meningitidis).
    return _from_checkm2_only(
        group_dir / "20251118_checkM2_quality_report.tsv",
        "Neisseria commensals",
    )


def _normalize_filename(s: pd.Series) -> pd.Series:
    """Strip directory and .fasta extension so campy assembly-stats and quality_report join cleanly."""
    return (
        s.astype(str)
        .str.replace(r"^.*/", "", regex=True)
        .str.replace(r"\.fasta$", "", regex=True)
    )


def _from_campy(asm_path: Path, qual_path: Path, species: str, contig_col: str) -> pd.DataFrame:
    """campy_coli / campy_jejuni: quality_report has Completeness_Specific + Genome_Size etc.
    Assembly-stats supplies contig count and longest contig (CheckM2 file here lacks Total_Contigs
    / Max_Contig_Length). Filenames are inconsistent — strip path + .fasta to merge."""
    asm = pd.read_csv(asm_path, sep="\t")
    qual = pd.read_csv(qual_path, sep="\t")
    asm["_key"] = _normalize_filename(asm["filename"])
    qual["_key"] = _normalize_filename(qual["Name"])
    merged = asm.merge(qual, on="_key", how="inner", suffixes=("_asm", "_qual"))
    if len(merged) < min(len(asm), len(qual)):
        log.warning(
            "%s: merge dropped rows (asm=%d qual=%d merged=%d)",
            species, len(asm), len(qual), len(merged),
        )
    out = pd.DataFrame({
        "Sample": merged["_key"],
        "N50": merged["N50"],                              # from assembly-stats
        "number": merged[contig_col],                      # No_contigs (coli) or number (jejuni)
        "longest": merged["longest"],
        "GC_Content": merged["GC_Content"].astype(float) * 100.0,
        "Completeness_Specific": merged["Completeness_Specific"].astype(float),
        "Contamination": merged["Contamination"].astype(float),
        "Total_Coding_Sequences": merged["Total_Coding_Sequences"],
        "Genome_Size": merged["Genome_Size"],
        "total_length": merged["total_length"],  # from assembly-stats
        "species_sylph": species,
    })
    return out


def load_campy_coli(group_dir: Path) -> pd.DataFrame:
    return _from_campy(
        group_dir / "Assembly-stats_Ccoli_PT_Human_2024-25.tsv",
        group_dir / "quality_report_Ccoli_PT_Human_2024-25.tsv",
        "Campylobacter coli",
        contig_col="No_contigs",
    )


def load_campy_jejuni(group_dir: Path) -> pd.DataFrame:
    return _from_campy(
        group_dir / "Assembly-stats_Cjejuni_PT_2023-2025.tsv.txt",
        group_dir / "quality_report_Cjejuni_PT_2023-2025.tsv.txt",
        "Campylobacter jejuni",
        contig_col="number",
    )


GROUPS = [
    ("axylosoxidans", load_axylosoxidans, "Achromobacter xylosoxidans"),
    ("campy_coli", load_campy_coli, "Campylobacter coli"),
    ("campy_jejuni", load_campy_jejuni, "Campylobacter jejuni"),
    ("neiserria/N_commensals", load_n_commensals, "Neisseria commensals"),
    ("neiserria/Ngo_for_Qualibact", load_ngo, "Neisseria gonorrhoeae"),
]


def build_merged(feedback_dir: Path) -> pd.DataFrame:
    frames = []
    for rel, loader, species in GROUPS:
        group_dir = feedback_dir / rel
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Group directory missing: {group_dir}")
        df = loader(group_dir)
        before = len(df)
        df = df.dropna(subset=[c for c in REQUIRED_COLS if c not in ("Sample",)])
        log.info("%-32s %s rows=%d (dropped %d with NaN metrics)", species, rel, len(df), before - len(df))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in merged.columns]
    if missing:
        raise RuntimeError(f"Output missing required columns: {missing}")
    return merged[REQUIRED_COLS]


def write_merged(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)
    log.info("Wrote %d rows to %s", len(df), path)


COMMENSAL_QUAST_REL = "neiserria/N_commensals/20251117_SRA_pregocomcom_quast.csv"
COMMENSAL_SPECIES_LABEL = "Neisseria commensals"
# Pathogens to drop from the commensal RefSeq fetch — gono is its own group, meningitidis
# is patently not a commensal. Match by substring, case-insensitive.
COMMENSAL_REFSEQ_PATHOGEN_EXCLUDES = ("gonorrhoeae", "meningitidis")


def commensal_constituent_species(feedback_dir: Path) -> list[str]:
    """Unique 'Species name' values from the commensal QUAST file (BOM stripped), with the
    obvious pathogens (gonorrhoeae, meningitidis) filtered out of the RefSeq fetch list."""
    quast = pd.read_csv(feedback_dir / COMMENSAL_QUAST_REL, encoding="utf-8-sig")
    species = (
        quast["Species name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    kept = []
    dropped = []
    for s in species:
        if not s:
            continue
        low = s.lower()
        if any(pat in low for pat in COMMENSAL_REFSEQ_PATHOGEN_EXCLUDES):
            dropped.append(s)
        else:
            kept.append(s)
    if dropped:
        log.info("Excluding %d pathogen species from commensal RefSeq fetch: %s",
                 len(dropped), sorted(dropped))
    return sorted(kept)


def prefetch_commensal_refseq(feedback_dir: Path, output_dir: Path, force: bool = False) -> None:
    """Stitch RefSeq summaries for every species named in the commensal file into a single
    cached '<species_dir>/Neisseria commensals.json'. The engine's run_datasets_summary will
    then short-circuit on the cached file instead of querying NCBI for the made-up taxon."""
    species_dir = output_dir / COMMENSAL_SPECIES_LABEL.replace(" ", "_")
    species_dir.mkdir(parents=True, exist_ok=True)
    cached_path = species_dir / f"{COMMENSAL_SPECIES_LABEL}.json"
    if cached_path.exists() and cached_path.stat().st_size > 0 and not force:
        log.info("Commensal refseq cache already present at %s — leaving as-is", cached_path)
        return

    constituents = commensal_constituent_species(feedback_dir)
    log.info("Fetching RefSeq summaries for %d constituent species: %s",
             len(constituents), constituents)

    combined_reports: list[dict] = []
    for sp in constituents:
        try:
            summary = run_datasets_summary(
                taxon=sp,
                outdir=str(species_dir),
                assembly_source="RefSeq",
                assembly_level="complete",
            )
        except Exception as exc:  # noqa: BLE001 - any datasets failure should not abort the whole run
            log.warning("Refseq fetch failed for %s: %s", sp, exc)
            continue
        if not summary or "reports" not in summary:
            log.warning("No RefSeq reports for %s", sp)
            continue
        combined_reports.extend(summary["reports"])
        log.info("  %-40s %d reports", sp, len(summary["reports"]))

    cached_path.write_text(json.dumps({"reports": combined_reports}))
    log.info("Wrote stitched commensal RefSeq cache (%d reports) to %s",
             len(combined_reports), cached_path)


def write_dotenv(env_path: Path, merged_tsv: Path, output_dir: Path) -> None:
    # Pin to the running interpreter so prepare's os.path.abspath(python_path) writes a real
    # executable into the SLURM scripts (the default "python3" gets resolved against the
    # repo root and produces a non-existent path). Same for the datasets binary.
    python_path = Path(sys.executable).resolve()
    bin_dir = (REPO_ROOT / "bin").resolve()
    env_path.write_text(
        f"QUALIBACT_MERGED_METADATA={merged_tsv}\n"
        f"QUALIBACT_OUTPUT_DIR={output_dir}\n"
        f"QUALIBACT_PYTHONPATH={python_path}\n"
        f"QUALIBACT_BIN={bin_dir}\n"
    )
    log.info("Wrote env file %s", env_path)


def run_qualibact(
    env_path: Path,
    output_dir: Path,
    species_list: list[str],
    submit: bool,
    feedback_dir: Path,
) -> None:
    runner = REPO_ROOT / "qualibact-run.py"
    python = sys.executable

    prefetch_commensal_refseq(feedback_dir, output_dir)

    log.info("Running prepare with %s", env_path)
    subprocess.run(
        [python, str(runner), "prepare", "--dotenv-path", str(env_path)]
        + (["--submit"] if submit else []),
        check=True,
        cwd=str(REPO_ROOT),
    )

    if submit:
        log.info("Submitted SLURM jobs via prepare; skipping inline calculate")
        return

    for species in species_list:
        species_dir = output_dir / species.replace(" ", "_").replace("/", "_")
        if not species_dir.is_dir():
            log.warning("Species dir not created by prepare: %s — skipping", species_dir)
            continue
        log.info("Running calculate for %s", species)
        subprocess.run(
            [python, str(runner), "calculate", species, str(species_dir),
             "--dotenv-path", str(env_path)],
            check=True,
            cwd=str(REPO_ROOT),
        )


TIDY_METRICS_OF_INTEREST = [
    "N50",
    "no_of_contigs",
    "GC_Content",
    "Genome_Size",
    "Completeness",
    "Contamination",
    "Total_Coding_Sequences",
]


def _round_metric(metric: str, lower, upper):
    """Apply the per-metric rounding rules from qualibact/summary_util.py tidy_summary
    (N50, Completeness, Contamination, Total_Coding_Sequences, no_of_contigs,
    GC_Content, Genome_Size). For GC we accept either fractional or percent inputs
    because the engine mixes both into MY_LOWER/MY_UPPER — anything <2 is treated as
    a fraction and rescaled. Returns (lower, upper) as int or '' sentinels."""
    lo = float(lower)
    hi = float(upper)
    if metric == "N50":
        return int(np.floor(lo / 1000) * 1000), ""
    if metric == "Completeness":
        return int(np.floor(lo)), ""
    if metric == "Contamination":
        return "", int(np.ceil(hi))
    if metric == "Total_Coding_Sequences":
        return int(np.floor(lo / 100) * 100), int(np.ceil(hi / 100) * 100)
    if metric == "no_of_contigs":
        return "", int(np.ceil(hi / 10) * 10)
    if metric == "GC_Content":
        lo_pct = lo * 100 if lo < 2 else lo
        hi_pct = hi * 100 if hi < 2 else hi
        return int(np.floor(lo_pct)), int(np.ceil(hi_pct))
    if metric == "Genome_Size":
        return int(np.floor(lo / 100000) * 100000), int(np.ceil(hi / 100000) * 100000)
    return lower, upper  # unknown metric: pass through unchanged


def make_tidy_summary(output_dir: Path, dest: Path) -> None:
    """Aggregate every species' selected_summary.csv into a single tidy filtered_metrics.csv
    with rounded lower/upper bounds. Mirrors qualibact/summary_util.py tidy_summary, but
    produced standalone so the engine's summary step doesn't need to run."""
    frames = []
    for species_dir in sorted(output_dir.iterdir()):
        sel = species_dir / "selected_summary.csv"
        if not sel.is_file():
            continue
        df = pd.read_csv(sel)
        df["species"] = species_dir.name.replace("_", " ")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No selected_summary.csv files found under {output_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["metric"] = combined["metric"].replace({
        "number": "no_of_contigs",
        "Completeness_Specific": "Completeness",
    })
    combined = combined[combined["metric"].isin(TIDY_METRICS_OF_INTEREST)]

    rounded = combined.apply(
        lambda r: pd.Series(_round_metric(r["metric"], r["MY_LOWER"], r["MY_UPPER"]),
                            index=["lower_bounds", "upper_bounds"]),
        axis=1,
    )
    tidy = combined[["species", "metric"]].copy()
    tidy[["lower_bounds", "upper_bounds"]] = rounded

    dest.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(dest, index=False)
    log.info("Wrote tidy summary (%d rows, %d species) to %s",
             len(tidy), tidy["species"].nunique(), dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    p.add_argument("--merged-tsv", type=Path, default=DEFAULT_MERGED_TSV,
                   help="Path for the assembled merged TSV (.tsv or .tsv.gz).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="QUALIBACT_OUTPUT_DIR for the run.")
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / ".qualibact.feedback.env",
                   help="Dotenv file written for the qualibact CLI invocations.")
    p.add_argument("--run", action="store_true", help="Invoke qualibact-run.py prepare + calculate after building.")
    p.add_argument("--submit", action="store_true", help="Pass --submit to prepare (SLURM array). Implies --run.")
    p.add_argument("--build-only", action="store_true", help="Only write the merged TSV; skip env + run.")
    p.add_argument("--prefetch-commensal-refseq", action="store_true",
                   help="Only (re)build the stitched RefSeq cache for Neisseria commensals.")
    p.add_argument("--force-prefetch", action="store_true",
                   help="Overwrite an existing commensal RefSeq cache.")
    p.add_argument("--make-tidy-summary", action="store_true",
                   help="Skip everything else and aggregate per-species selected_summary.csv "
                        "into a single rounded filtered_metrics.csv under --output-dir.")
    p.add_argument("--tidy-output", type=Path,
                   help="Path for the tidy summary CSV (default: <output-dir>/filtered_metrics.csv).")
    args = p.parse_args()

    if args.make_tidy_summary:
        dest = args.tidy_output or (args.output_dir.resolve() / "filtered_metrics.csv")
        make_tidy_summary(args.output_dir.resolve(), dest)
        return

    if args.prefetch_commensal_refseq:
        prefetch_commensal_refseq(args.feedback_dir, args.output_dir.resolve(),
                                  force=args.force_prefetch)
        return

    merged = build_merged(args.feedback_dir)
    write_merged(merged, args.merged_tsv)

    if args.build_only:
        return

    write_dotenv(args.env_file, args.merged_tsv.resolve(), args.output_dir.resolve())

    if args.run or args.submit:
        run_qualibact(
            args.env_file,
            args.output_dir.resolve(),
            species_list=[s for _, _, s in GROUPS],
            submit=args.submit,
            feedback_dir=args.feedback_dir,
        )
    else:
        log.info("Built merged TSV. Re-run with --run to invoke the qualibact pipeline.")


if __name__ == "__main__":
    main()

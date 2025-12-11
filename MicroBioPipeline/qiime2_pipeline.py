#!/usr/bin/env python3
"""
qiime2_pl.py

A split QIIME2 pipeline that cleanly separates:
 - manifest generation
 - import + quality check (create demux artifact and demux_summary.qzv)
 - denoising (DADA2) + taxonomy + tree + diversity (resumes from demux artifact)

Manifest generation now supports two naming conventions for paired-end or single-end files:
  - Default ('_1'/'_2'/'_4' at name end): detects 'forward', 'reverse', or single-end reads based on trailing digit after underscore.
  - R pattern ('R1'/'R2' in file name): detects 'forward' and 'reverse' reads based on presence of R1 or R2 within the base name (case-insensitive).

Usage examples:

1) Create manifest (default pattern for _1/_2):
   python qiime2_pl.py make-manifest \
     --fastq_dir /path/to/fastqs \
     --manifest_type paired \
     --manifest_output manifest.csv \
     --read_pattern default

2) Create manifest (R-pattern for R1/R2):
   python qiime2_pl.py make-manifest \
     --fastq_dir /path/to/fastqs \
     --manifest_type paired \
     --manifest_output manifest.csv \
     --read_pattern R

3) Import and generate quality summary (inspect demux_summary.qzv in view.qiime2.org):
   python qiime2_pl.py import-qc \
     --manifest ./Ovarian/fastq/Asangba_2023/manifest_Asangba_2023.tsv --paired \
     --demux_out ./Ovarian/fastq/Asangba_2023/Results/1_Quality/demux.qza

4) After inspecting demux_summary.qzv and choosing trimming/truncation values, run denoise+rest:
   python qiime2_pl.py run \
     --demux ./Ovarian/fastq/Asangba_2023/Results/1_Quality/demux.qza \
     --classifier /home/sabrina.tamburini/WorkSpace/TaxonomyClassifier/silva-138-99-nb-classifier.qza \
     --metadata ./Ovarian/InfoFile/metadata_Asangba_2023.tsv \
     --paired \
     --trim_left_f 0 --trim_left_r 0 \
     --trunc_len_f 140 --trunc_len_r 120 \
     --depth 10000

Arguments of make-manifest:
  --fastq_dir             The directory containing FASTQ files.
  --manifest_type         'paired', 'forward', or 'reverse'.
  --manifest_output       Output file name for the manifest (CSV or TSV).
  --manifest_delimiter    Use ',' for CSV, '\t' for TSV.
  --read_pattern          'default' for _1/_2/_4 (QIIME2 standard), 'R' for R1/R2 in file name (Illumina standard).

"""

import os
import csv
import glob
import argparse
import re
import qiime2
from qiime2.plugins import demux, dada2, feature_classifier, feature_table, alignment, phylogeny, diversity

def _ensure_dir_for(path):
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    return d

def _qzv_path_next_to(qza_path, default_name):
    qza_dir = os.path.dirname(os.path.abspath(qza_path)) or '.'
    return os.path.join(qza_dir, default_name)

##########################
# Manifest Preparation
##########################

def create_manifest(folder_path, output_file, manifest_type, delimiter=",", read_pattern="default"):
    """
    Create a QIIME2 manifest file from a directory of FASTQ files.
    Arguments:
      folder_path: Directory containing FASTQ files.
      output_file: Output file for manifest.
      manifest_type: 'paired', 'forward', 'reverse'
      delimiter: CSV delimiter (',' or '\t')
      read_pattern: 'default' (trailing _1/_2/_4) or 'R' (R1/R2 in name)

    Detection logic:
      - If read_pattern == 'default':
          Remove .gz/.fastq. If base name ends in _1/_2/_4, detects as paired or single-end, otherwise single-end.
      - If read_pattern == 'R':
          Uses case-insensitive search for R1/R2 in file name (by regex). Everything else treated as single-end.
    """
    samples = {}

    fastq_files = glob.glob(os.path.join(folder_path, "*.fastq*"))
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {folder_path}")

    for file in fastq_files:
        filename = os.path.basename(file)

        # strip .gz if present
        name = filename[:-3] if filename.endswith('.gz') else filename
        # strip .fastq
        if name.endswith('.fastq'):
            name_core = name[:-6]
        else:
            print(f"[WARN] Skipping non-fastq file: {filename}")
            continue

        sample_id = None
        direction = None

        if read_pattern.lower() == 'r':
            # Detect R1/R2 using regex, handle -R1, _R1, .R1 etc.
            # Only match R1/R2 followed by optional spaces, dots, underscores, or - at end or before file extension
            if re.search(r'R1([._\-]|$)', name_core, re.IGNORECASE):
                sample_id = re.sub(r'R1([._\-]|$)', '', name_core, flags=re.IGNORECASE).strip('_-.')
                direction = "forward"
            elif re.search(r'R2([._\-]|$)', name_core, re.IGNORECASE):
                sample_id = re.sub(r'R2([._\-]|$)', '', name_core, flags=re.IGNORECASE).strip('_-.')
                direction = "reverse"
            else:
                # treat as single-end/forward only
                sample_id = name_core
                direction = "forward"
        else:  # default: trailing _1/_2/_4
            if '_' in name_core:
                sample_part, last_token = name_core.rsplit('_', 1)
                if last_token in ('1', '2', '4'):
                    sample_id = sample_part
                    read_code = last_token
                else:
                    sample_id = name_core
                    read_code = None
            else:
                sample_id = name_core
                read_code = None

            if 'read_code' in locals():  # only exists for default pattern
                if read_code == '1':
                    direction = "forward"
                elif read_code == '2':
                    direction = "reverse"
                elif read_code == '4' or read_code is None:
                    direction = "forward"
                else:
                    print(f"[WARN] Unrecognized read code in file {filename}; skipping.")
                    continue

        if sample_id is not None and direction is not None:
            samples.setdefault(sample_id, {})[direction] = os.path.abspath(file)
        else:
            print(f"[WARN] Could not detect sample_id/direction for file {filename}; skipping.")
            continue

    with open(output_file, "w", newline="") as manifest:
        writer = csv.writer(manifest, delimiter=delimiter)

        if manifest_type == "paired":
            writer.writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])
        else:
            writer.writerow(["sample-id", "absolute-filepath"])

        for sample_id, files in sorted(samples.items()):
            has_fwd = "forward" in files
            has_rev = "reverse" in files

            if manifest_type == "paired":
                if has_fwd and has_rev:
                    writer.writerow([sample_id, files["forward"], files["reverse"]])
                else:
                    print(f"[WARN] sample '{sample_id}' missing forward or reverse read; not included in paired manifest.")
            elif manifest_type == "forward":
                if has_fwd:
                    writer.writerow([sample_id, files["forward"]])
                else:
                    print(f"[WARN] sample '{sample_id}' has no forward read.")
            elif manifest_type == "reverse":
                if has_rev:
                    writer.writerow([sample_id, files["reverse"]])
                else:
                    print(f"[WARN] sample '{sample_id}' has no reverse read.")
    print(f"[INFO] Manifest file generated: {output_file}")
    print(f"[INFO] Pattern used for detection: {'R1/R2 in filename' if read_pattern.lower() == 'r' else '_1/_2/_4 trailing'}")

##########################
# Import + QC (stop for inspection)
##########################

def import_and_qc(manifest, paired, demux_out="demux.qza", delimiter=","):
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    demux_out_abs = os.path.abspath(demux_out)
    _ensure_dir_for(demux_out_abs)

    print("[INFO] Importing FASTQ data using manifest...")
    if paired:
        view_type = 'PairedEndFastqManifestPhred33V2'
        imp_type = 'SampleData[PairedEndSequencesWithQuality]'
    else:
        view_type = 'SingleEndFastqManifestPhred33V2'
        imp_type = 'SampleData[SequencesWithQuality]'

    demuxed = qiime2.Artifact.import_data(
        imp_type, manifest, view_type=view_type
    )

    demuxed.save(demux_out_abs)
    print(f"[INFO] Demux artifact saved to: {demux_out_abs}")

    print("[INFO] Generating demux summary visualization (demux_summary.qzv) next to the demux qza...")
    viz = demux.visualizers.summarize(demuxed)
    qzv_path = _qzv_path_next_to(demux_out_abs, "demux_summary.qzv")
    viz.visualization.save(qzv_path)
    print(f"[INFO] demux_summary.qzv created at: {qzv_path}")
    print("[ACTION] Choose trimming/truncation values based on the quality plots, then re-run the 'run' command.")

##########################
# Denoise + downstream analysis (resume)
##########################

def run_denoise_and_rest(demux_artifact_path, paired, classifier, metadata,
                         trim_left_f, trim_left_r, trunc_len_f, trunc_len_r,
                         depth, permanova, permanova_column):
    if not os.path.exists(demux_artifact_path):
        raise FileNotFoundError(f"Demux artifact not found: {demux_artifact_path}")

    print("[INFO] Loading demux artifact...")
    demuxed = qiime2.Artifact.load(demux_artifact_path)

    # DADA2
    print("[INFO] Running DADA2...")
    if paired:
        denoise = dada2.methods.denoise_paired(
            demultiplexed_seqs=demuxed,
            trim_left_f=trim_left_f,
            trim_left_r=trim_left_r,
            trunc_len_f=trunc_len_f,
            trunc_len_r=trunc_len_r
        )
    else:
        denoise = dada2.methods.denoise_single(
            demultiplexed_seqs=demuxed,
            trim_left=trim_left_f,
            trunc_len=trunc_len_f
        )

    table_qza = os.path.abspath('table.qza')
    repseqs_qza = os.path.abspath('rep-seqs.qza')
    denoising_stats_qza = os.path.abspath('denoising-stats.qza')

    print("[INFO] Saving DADA2 outputs: table.qza, rep-seqs.qza, denoising-stats.qza")
    denoise.table.save(table_qza)
    denoise.representative_sequences.save(repseqs_qza)
    denoise.denoising_stats.save(denoising_stats_qza)

    # Taxonomy assignment
    if classifier:
        if not os.path.exists(classifier):
            raise FileNotFoundError(f"Classifier artifact not found: {classifier}")
        print("[INFO] Classifying taxonomy with classifier:", classifier)
        classifier_art = qiime2.Artifact.load(classifier)
        taxonomy = feature_classifier.methods.classify_sklearn(
            reads=denoise.representative_sequences,
            classifier=classifier_art
        )
        taxonomy_qza = os.path.abspath('taxonomy.qza')
        taxonomy.classification.save(taxonomy_qza)
        print(f"[INFO] Taxonomy saved to {taxonomy_qza}")
    else:
        print("[WARN] No classifier provided; skipping taxonomy assignment.")

    # Feature table summary (save qzv next to table.qza)
    if metadata:
        if not os.path.exists(metadata):
            raise FileNotFoundError(f"Metadata file not found: {metadata}")
        print("[INFO] Generating feature-table summary (table.qzv)...")
        md = qiime2.Metadata.load(metadata)
        ft_viz = feature_table.visualizers.summarize(
            denoise.table,
            sample_metadata=md
        )
        table_qzv = _qzv_path_next_to(table_qza, "table.qzv")
        ft_viz.visualization.save(table_qzv)
        print(f"[INFO] table.qzv created at: {table_qzv}")
    else:
        md = None
        print("[WARN] No metadata provided; skipping feature-table summary (table.qzv)")

    print("[INFO] Calculating relative frequency table (rel-table.qza)...")
    rel_freq = feature_table.methods.relative_frequency(denoise.table)
    rel_table_qza = os.path.abspath('rel-table.qza')
    rel_freq.relative_frequency_table.save(rel_table_qza)

    print("[INFO] Generating phylogenetic tree (rooted-tree.qza)...")
    aligned = alignment.methods.mafft(
        sequences=denoise.representative_sequences
    )
    masked = alignment.methods.mask(aligned.alignment)
    tree = phylogeny.methods.fasttree(masked.masked_alignment)
    rooted = phylogeny.methods.midpoint_root(tree.tree)
    rooted_tree_qza = os.path.abspath('rooted-tree.qza')
    rooted.rooted_tree.save(rooted_tree_qza)

    if metadata:
        print("[INFO] Running diversity core-metrics-phylogenetic...")
        core_metrics = diversity.pipelines.core_metrics_phylogenetic(
            table=denoise.table,
            phylogeny=rooted.rooted_tree,
            metadata=md,
            sampling_depth=depth
        )

        faith_qza = os.path.abspath('faith_pd_vector.qza')
        unweighted_unifrac_qza = os.path.abspath('unweighted_unifrac.qza')
        unweighted_unifrac_pcoa_qza = os.path.abspath('unweighted_unifrac_pcoa.qza')
        beta_group_sig_qzv = os.path.abspath('beta-group-significance.qzv')

        print("[INFO] Saving diversity outputs...")
        core_metrics.faith_pd_vector.save(faith_qza)
        core_metrics.unweighted_unifrac_distance_matrix.save(unweighted_unifrac_qza)
        core_metrics.unweighted_unifrac_pcoa_results.save(unweighted_unifrac_pcoa_qza)

        beta_qzv_path = _qzv_path_next_to(unweighted_unifrac_qza, "beta-group-significance.qzv")
        core_metrics.beta_group_significance.visualization.save(beta_qzv_path)
        print(f"[INFO] beta-group-significance.qzv created at: {beta_qzv_path}")
    else:
        print("[WARN] Metadata not provided; skipping diversity core metrics.")
        core_metrics = None

    if permanova and permanova_column:
        if not metadata:
            raise ValueError("PERMANOVA requested but no metadata provided.")
        print(f"[INFO] Performing PERMANOVA on column '{permanova_column}'...")
        try:
            distance_matrix = core_metrics.unweighted_unifrac_distance_matrix
        except Exception:
            raise RuntimeError("Distance matrix required for PERMANOVA not found. Run core metrics first.")
        metadata_col = md.get_column(permanova_column)
        permanova_viz = diversity.visualizers.beta_group_significance(
            distance_matrix=distance_matrix,
            metadata=metadata_col,
            method="permanova",
            pairwise=True
        )
        permanova_qzv = _qzv_path_next_to(unweighted_unifrac_qza, f"permanova-{permanova_column}.qzv")
        permanova_viz.visualization.save(permanova_qzv)
        print(f"[INFO] PERMANOVA result saved to {permanova_qzv}")

    print("[DONE] Pipeline finished. Visualize .qzv files at https://view.qiime2.org")

##########################
# CLI (argparse)
##########################

def main():
    parser = argparse.ArgumentParser(
        description="Split QIIME2 pipeline: manifest -> import+QC -> denoise+rest"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # make-manifest subcommand
    p_manifest = subparsers.add_parser("make-manifest", help="Generate QIIME2 manifest from FASTQ folder")
    p_manifest.add_argument('--fastq_dir', type=str, required=True, help='Directory of FASTQ files for manifest')
    p_manifest.add_argument('--manifest_type', choices=['paired', 'forward', 'reverse'],
                            default='paired', help='Manifest type')
    p_manifest.add_argument('--manifest_delimiter', choices=[',', '\t'], default=',',
                            help='Delimiter for manifest file (default: csv)')
    p_manifest.add_argument('--manifest_output', type=str, default='manifest.csv',
                            help='Output manifest filename (default: manifest.csv)')
    p_manifest.add_argument('--read_pattern', choices=['default', 'R'], default='default',
                            help="Pattern for read detection: 'default' for _1/_2/_4, 'R' for R1/R2 in filenames.")

    # import-qc subcommand
    p_import = subparsers.add_parser("import-qc", help="Import FASTQ using manifest and generate demux_summary.qzv (stop for inspection)")
    p_import.add_argument('--manifest', type=str, required=True, help='Manifest file (csv or tsv) for import')
    p_import.add_argument('--paired', action='store_true', help='Set for paired-end reads')
    p_import.add_argument('--demux_out', type=str, default='demux.qza', help='Output demux artifact filename')
    p_import.add_argument('--manifest_delimiter', choices=[',', '\t'], default='\t', help='Delimiter used in manifest')

    # run subcommand (resume from demux)
    p_run = subparsers.add_parser("run", help="Run DADA2 and downstream analysis from a demux artifact")
    p_run.add_argument('--demux', type=str, required=True, help='Demux artifact (.qza) created by import-qc')
    p_run.add_argument('--classifier', type=str, help='QIIME2 .qza classifier for taxonomy (optional)')
    p_run.add_argument('--metadata', type=str, help='QIIME2 sample metadata .tsv (required for many downstream steps)')
    p_run.add_argument('--paired', action='store_true', help='Set for paired-end reads')
    p_run.add_argument('--trim_left_f', type=int, default=0, help='Trim left bases (forward)')
    p_run.add_argument('--trim_left_r', type=int, default=0, help='Trim left bases (reverse)')
    p_run.add_argument('--trunc_len_f', type=int, default=240, help='Truncate length (forward)')
    p_run.add_argument('--trunc_len_r', type=int, default=200, help='Truncate length (reverse)')
    p_run.add_argument('--depth', type=int, default=10000, help='Sampling depth for diversity')
    p_run.add_argument('--permanova', action='store_true', help='Perform PERMANOVA test')
    p_run.add_argument('--permanova_column', type=str, help='Metadata column for PERMANOVA')

    args = parser.parse_args()

    try:
        if args.command == "make-manifest":
            create_manifest(os.path.abspath(args.fastq_dir), args.manifest_output, args.manifest_type,
                            delimiter=args.manifest_delimiter, read_pattern=args.read_pattern)
        elif args.command == "import-qc":
            import_and_qc(args.manifest, args.paired, demux_out=args.demux_out, delimiter=args.manifest_delimiter)
        elif args.command == "run":
            run_denoise_and_rest(
                demux_artifact_path=args.demux,
                paired=args.paired,
                classifier=args.classifier,
                metadata=args.metadata,
                trim_left_f=args.trim_left_f,
                trim_left_r=args.trim_left_r,
                trunc_len_f=args.trunc_len_f,
                trunc_len_r=args.trunc_len_r,
                depth=args.depth,
                permanova=args.permanova,
                permanova_column=args.permanova_column
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()
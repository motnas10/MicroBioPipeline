#!/usr/bin/env python3
"""
QIIME2_pipeline. py

Split QIIME2 pipeline for 16S analysis:  
 1.  Manifest generation (supports both trailing _1/_2/_4 and R1/R2 patterns)
 2. Import & QC (stop to inspect demux_summary.qzv and choose trimming/truncation/rarefaction depth)
 3. Denoise (DADA2), taxonomy, taxonomy-based feature filtering, optional rarefaction, tree, and customizable diversity metrics
 4. Convert QZA to QZV (generate visualizations from artifacts)

Behavior changes (important):
- All produced . qza/. qzv outputs are stored under a `results` folder located in the main folder
  inferred from the provided fastq/manifest/demux path (e.g. /path/to/fastqs/results/01_demux, .. .).
- The `results` folder contains numbered subfolders for each analysis step (01_demux, 02_denoise, .. .).
- The script no longer attempts to create PCoA/Emperor visualizations (some qiime2 installs
  do not expose diversity. visualizers.pcoa). The pipeline saves beta distance matrices (. qza).
  If you need PCoA/Emperor, generate them manually (example CLI commands provided below).

Usage examples (ready to run):

---------------------------------------------------------------------------------------------------
1) Create a paired manifest (default underscore _1/_2 pattern), write TSV manifest:  
python QIIME2_pipeline. py make-manifest \
  --fastq_dir /path/to/fastqs \
  --manifest_type paired \
  --manifest_output /path/to/fastqs/manifest.tsv \
  --manifest_delimiter '\t' \
  --read_pattern default

2) Create a manifest when your reads are named with R1/R2:
python QIIME2_pipeline.py make-manifest \
  --fastq_dir ./fastq_SanCamillo/fastqPaired \
  --manifest_type paired \
  --manifest_output ./fastq_SanCamillo/fastqPaired/manifest.tsv \
  --manifest_delimiter '\t' \
  --read_pattern R

python Executor.py ./QIIME2_pipeline.py --args make-manifest \
  --fastq_dir ./Vaginal/fastq/Jacobson_2021 \
  --manifest_type paired \
  --manifest_output ./Vaginal/fastq/Jacobson_2021/Jacobson_2021_manifest.tsv \
  --read_pattern default

---------------------------------------------------------------------------------------------------
3) Import FASTQ and create demux summary visualization:
python QIIME2_pipeline.py import-qc \
  --manifest ./fastq_SanCamillo/fastqPaired/manifest.tsv \
  --paired \
  --demux_out paired-demux.qza \
  --manifest_delimiter '\t'

Note: demux. qza and demux_summary.qzv will be written to:  
  /path/to/fastqs/results/01_demux/

Inspect demux_summary.qzv in https://view.qiime2.org to pick trim/trunc values.  

python Executor.py ./QIIME2_pipeline.py --no-push --args import-qc \
  --manifest ./Vaginal/fastq/Jacobson_2021/Jacobson_2021_manifest.tsv \
  --paired \
  --demux_out paired-demux.qza


---------------------------------------------------------------------------------------------------
4) Run denoise, taxonomy, filtering, diversity (example):
python QIIME2_pipeline.py run \
  --demux ./Vaginal/fastq/Jacobson_2021/01_demux/paired-demux.qza \
  --classifier ./Classifiers/SilvaClassifier/silva-138-99-nb-classifier.qza \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv \
  --paired \
  --trim_left_f 19 --trim_left_r 20 \
  --trunc_len_f 200 --trunc_len_r 160 \
  --min_samples 2 --min_frequency 10 \
  --alpha_metrics shannon,faith_pd,observed_features \
  --beta_metrics jaccard,braycurtis,unweighted_unifrac,weighted_unifrac \
  --depth 20864 \
  --permanova \
  --permanova_column TreatmentGroup


python Executor.py ./QIIME2_pipeline.py --no-push --args run \
  --demux ./Vaginal/fastq/Jacobson_2021/results/01_demux/paired-demux.qza \
  --classifier /home/mattiasantin/WorkSpace/QIIME2Framework/Classifiers/SilvaClassifier/silva-138-99-nb-classifier.qza \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv \
  --paired \
  --trim_left_f 19 --trim_left_r 20 \
  --trunc_len_f 200 --trunc_len_r 160 \
  --alpha_metrics shannon,faith_pd,observed_features \
  --beta_metrics jaccard,braycurtis,unweighted_unifrac,weighted_unifrac \
  --depth 10 \
  --permanova \
  --permanova_column Tumor

---------------------------------------------------------------------------------------------------
5) Convert QZA artifacts to QZV visualizations: 

# Convert a single file
python QIIME2_pipeline.py convert-to-qzv \
  --input ./Vaginal/fastq/Jacobson_2021/results/02_denoise/table.qza \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv

# Convert all files in a specific directory
python QIIME2_pipeline.py convert-to-qzv \
  --input_dir ./Vaginal/fastq/Jacobson_2021/results/07_diversity \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv \
  --pattern "alpha_*. qza"

# Convert ALL . qza files recursively in the entire results folder
python QIIME2_pipeline.py convert-to-qzv \
  --input_dir ./Vaginal/fastq/Jacobson_2021/results \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv \
  --recursive

python Executor.py ./QIIME2_pipeline.py --no-push --args convert-to-qzv \
  --input_dir ./Vaginal/fastq/Jacobson_2021/results \
  --metadata ./Vaginal/InfoFile/metadata_Jacobson_2021.tsv \
  --recursive

---------------------------------------------------------------------------------------------------
Where outputs will be placed (example):
- results/02_denoise/:  table.qza, rep-seqs.qza, denoising-stats.qza, rep-seqs. qzv
- results/03_taxonomy/: taxonomy.qza, taxa-barplot.qzv (if metadata present)
- results/04_filtering/: table_filtered.qza, table_filtered.qzv
- results/05_rel_freq/: rel-table.qza, rel-table. qzv (if metadata)
- results/06_tree/: rooted-tree.qza
- results/07_diversity/:  alpha_*. qza, alpha_*_group_significance.qzv, beta_*.qza, permanova-*.qzv

Manual PCoA/Emperor (if required and supported by your qiime2 install):
# compute PCoA from distance matrix
qiime diversity pcoa \
  --i-distance-matrix /path/to/fastqs/results/07_diversity/beta_jaccard. qza \
  --o-pcoa /path/to/fastqs/results/07_diversity/beta_jaccard-pcoa.qza

# make Emperor ordination plot using the PCoA and your metadata
qiime emperor plot \
  --i-pcoa /path/to/fastqs/results/07_diversity/beta_jaccard-pcoa.qza \
  --m-metadata-file /path/to/fastqs/sample-metadata.tsv \
  --o-visualization /path/to/fastqs/results/07_diversity/beta_jaccard-emperor.qzv

If you prefer, I can add a guarded option to attempt PCoA/Emperor only when those visualizers exist.  
"""

import os
import csv
import glob
import argparse
import re
import qiime2
from qiime2.plugins import demux, dada2, feature_classifier, feature_table, alignment, phylogeny, diversity, taxa, metadata as q2_metadata

##########################
# Utility Functions
##########################

def _ensure_dir_for(path):
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    return d

def infer_fastq_root_from_path(path):
    """
    Heuristic to find the main fastq folder root from a given path (manifest, demux qza, etc).
    - If the path contains a 'results' path element, we take the parent before 'results' as root.  
    - Otherwise, we use the directory containing the path.  
    - If that directory contains . fastq files, we prefer it.
    - Fall back to current working directory.
    """
    if not path:  
        return os.getcwd()
    p = os.path.abspath(path)
    # If it's a file (existing or not), use its directory for inference
    if os.path.isfile(p) or os.path.splitext(p)[1]:
        base = os.path.dirname(p)
    else:
        base = p

    parts = base.split(os.sep)
    if 'results' in parts:
        idx = parts.index('results')
        root_candidate = os.sep.join(parts[:idx]) or os.sep
        return root_candidate

    # If the directory contains fastq files, take it
    fastqs = glob.glob(os.path.join(base, "*. fastq*"))
    if fastqs:
        return base

    # otherwise, try parent dirs up to root to find fastqs
    cur = base
    while True:
        cur_fastqs = glob.glob(os. path.join(cur, "*. fastq*"))
        if cur_fastqs:
            return cur
        parent = os.path. dirname(cur)
        if parent == cur or not parent:
            break
        cur = parent

    # fallback
    return base or os.getcwd()

def make_results_subdirs(root):
    """
    Create and return a dict of numbered subdirectories under root/results for the pipeline steps.
    """
    results_root = os.path.join(root, "results")
    dirs = {
        "demux": os.path.join(results_root, "01_demux"),
        "denoise":  os.path.join(results_root, "02_denoise"),
        "taxonomy": os.path.join(results_root, "03_taxonomy"),
        "filtering": os.path.join(results_root, "04_filtering"),
        "rel_freq": os.path.join(results_root, "05_rel_freq"),
        "tree": os.path.join(results_root, "06_tree"),
        "diversity": os.path.join(results_root, "07_diversity"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def safe_save_artifact(artifact, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    artifact.save(path)
    print(f"[INFO] Saved artifact: {path}")

##########################
# I.  Manifest Preparation
##########################

def create_manifest(folder_path, output_file, manifest_type, delimiter=",", read_pattern="default"):
    """
    Create a QIIME2 manifest file from a directory of FASTQ files.
    Supports trailing _1/_2/_4 and R1/R2 patterns.  
    """
    samples = {}
    fastq_files = glob.glob(os.path.join(folder_path, "*.fastq*"))
    if not fastq_files:
        raise FileNotFoundError(f"No FASTQ files found in {folder_path}")

    for file in fastq_files:
        filename = os.path.basename(file)
        # strip . gz if present
        name = filename[:-3] if filename.endswith('. gz') else filename
        # strip . fastq extension
        if name.endswith('.fastq'):
            name_core = name[:-6]
        else:
            print(f"[WARN] Skipping non-fastq file: {filename}")
            continue

        sample_id, direction = None, None
        if read_pattern. lower() == 'r':
            # match R1/R2 patterns
            if re.search(r'R1([._\-]|$)', name_core, re.IGNORECASE):
                sample_id = re.sub(r'R1([._\-]|$)', '', name_core, flags=re.IGNORECASE).strip('_-.')
                direction = "forward"
            elif re.search(r'R2([._\-]|$)', name_core, re.IGNORECASE):
                sample_id = re.sub(r'R2([._\-]|$)', '', name_core, flags=re.IGNORECASE).strip('_-.')
                direction = "reverse"
            else:
                sample_id = name_core
                direction = "forward"
        else:  # default _1/_2/_4 pattern
            if '_' in name_core:
                sample_part, last_token = name_core.rsplit('_', 1)
                if last_token in ('1', '2', '4'):
                    sample_id = sample_part
                    read_code = last_token
                else:
                    sample_id, read_code = name_core, None
            else:
                sample_id, read_code = name_core, None
            if 'read_code' in locals():
                if read_code == '1':
                    direction = "forward"
                elif read_code == '2':
                    direction = "reverse"
                elif read_code == '4' or read_code is None:
                    direction = "forward"
                else:
                    continue

        if sample_id is not None and direction is not None:
            samples. setdefault(sample_id, {})[direction] = os.path.abspath(file)

    # Ensure parent dir for manifest exists
    _ensure_dir_for(output_file)

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
            elif manifest_type == "forward":
                if has_fwd:
                    writer.writerow([sample_id, files["forward"]])
            elif manifest_type == "reverse": 
                if has_rev:
                    writer.writerow([sample_id, files["reverse"]])
    print(f"[INFO] Manifest file generated: {output_file}")

    # Also print where results would be placed if the user runs other steps
    root = infer_fastq_root_from_path(folder_path)
    print(f"[INFO] Inferred fastq root: {root}")
    print(f"[INFO] Results will be written under: {os.path.join(root, 'results')} when running subsequent steps")

##########################
# II. Import & QC
##########################

def import_and_qc(manifest, paired, demux_out="demux. qza", delimiter="\t"):
    """
    Import FASTQ using manifest and generate demux_summary.qzv for quality inspection.
    Outputs are written to results/01_demux under the inferred fastq root.
    """
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # Infer results root from manifest directory
    root = infer_fastq_root_from_path(manifest)
    dirs = make_results_subdirs(root)
    demux_dir = dirs["demux"]

    demux_out_abs = os.path.abspath(os.path.join(demux_dir, os.path.basename(demux_out)))
    os.makedirs(demux_dir, exist_ok=True)

    if paired:
        view_type = 'PairedEndFastqManifestPhred33V2'
        imp_type = 'SampleData[PairedEndSequencesWithQuality]'
    else:
        view_type = 'SingleEndFastqManifestPhred33V2'
        imp_type = 'SampleData[SequencesWithQuality]'

    print("[INFO] Importing reads using manifest...")
    demuxed = qiime2.Artifact. import_data(imp_type, manifest, view_type=view_type)
    demuxed.save(demux_out_abs)
    print(f"[INFO] Demux artifact saved to: {demux_out_abs}")

    print("[INFO] Generating demux summary visualization (demux_summary.qzv)...")
    viz = demux.visualizers.summarize(demuxed)
    qzv_path = os.path. join(demux_dir, "demux_summary.qzv")
    viz.visualization. save(qzv_path)
    print(f"[INFO] demux_summary.qzv created at: {qzv_path}")
    print("[ACTION] Review this file in view. qiime2.org to select trimming/truncation and rarefaction --depth for the next step.")

##########################
# III. Denoise, Taxonomy, Filtering, Tree, Diversity
##########################

def run_denoise_and_rest(
    demux_artifact_path, paired, classifier, metadata,
    trim_left_f, trim_left_r, trunc_len_f, trunc_len_r,
    depth, permanova, permanova_column,
    min_samples=2, min_frequency=10,
    alpha_metrics_arg=None, beta_metrics_arg=None
):
    # 1. Load demux
    if not os.path.exists(demux_artifact_path):
        raise FileNotFoundError(f"Demux artifact not found: {demux_artifact_path}")

    # Infer results root from demux artifact path
    root = infer_fastq_root_from_path(demux_artifact_path)
    dirs = make_results_subdirs(root)

    # Load metadata early so it's available for various steps
    md = None
    if metadata:
        if not os.path.exists(metadata):
            print(f"[WARN] Metadata file not found: {metadata}. Metadata-dependent steps will be skipped.")
            md = None
        else:
            try:
                md = qiime2.Metadata.load(metadata)
            except Exception as e:
                print(f"[WARN] Could not load metadata {metadata}: {e}")
                md = None

    demuxed = qiime2.Artifact.load(demux_artifact_path)

    # 2. DADA2 Denoising
    print("[INFO] Running DADA2 denoise...")
    if paired:
        denoise = dada2.methods.denoise_paired(
            demultiplexed_seqs=demuxed,
            trim_left_f=trim_left_f,
            trim_left_r=trim_left_r,
            trunc_len_f=trunc_len_f,
            trunc_len_r=trunc_len_r,
        )
    else:
        denoise = dada2.methods. denoise_single(
            demultiplexed_seqs=demuxed,
            trim_left=trim_left_f,
            trunc_len=trunc_len_f
        )

    # Save denoise outputs into 02_denoise
    denoise_dir = dirs["denoise"]
    table_qza = os.path.abspath(os.path.join(denoise_dir, 'table.qza'))
    repseqs_qza = os. path.abspath(os.path.join(denoise_dir, 'rep-seqs.qza'))
    denoising_stats_qza = os.path.abspath(os.path. join(denoise_dir, 'denoising-stats.qza'))

    denoise.table.save(table_qza)
    denoise.representative_sequences.save(repseqs_qza)
    denoise.denoising_stats.save(denoising_stats_qza)
    print(f"[INFO] Denoise artifacts saved to: {denoise_dir}")

    # Try to create a rep-seqs visualization (tabulate seqs)
    try:
        repseqs_viz = feature_table.visualizers.tabulate_seqs(sequences=denoise.representative_sequences)
        repseqs_qzv = os.path.join(denoise_dir, 'rep-seqs.qzv')
        repseqs_viz.visualization.save(repseqs_qzv)
        print(f"[INFO] rep-seqs.qzv created at: {repseqs_qzv}")
    except Exception as e:
        print(f"[WARN] Could not create rep-seqs.qzv: {e}")

    # 3. Assign Taxonomy
    taxonomy_qza = None
    if classifier:
        print("[INFO] Assigning taxonomy...")
        classifier_art = qiime2.Artifact. load(classifier)
        taxonomy = feature_classifier.methods.classify_sklearn(
            reads=denoise.representative_sequences, classifier=classifier_art
        )
        taxonomy_dir = dirs["taxonomy"]
        taxonomy_qza = os.path. abspath(os.path.join(taxonomy_dir, 'taxonomy.qza'))
        taxonomy.classification.save(taxonomy_qza)
        print(f"[INFO] Taxonomy artifact saved to: {taxonomy_qza}")

        # Try to make a taxa barplot if metadata and table are available
        if md:  
            try:
                bp = taxa.visualizers.barplot(table=denoise.table, taxonomy=taxonomy. classification, metadata=md)
                bp_qzv = os.path. join(taxonomy_dir, 'taxa-barplot.qzv')
                bp.visualization.save(bp_qzv)
                print(f"[INFO] taxa-barplot.qzv created at: {bp_qzv}")
            except Exception as e:  
                print(f"[WARN] Could not create taxa barplot: {e}")
        else:
            print("[WARN] Metadata not provided or missing - skipping taxa barplot creation.")
    else:
        print("[WARN] No classifier provided; skipping taxonomy assignment.")

    # 4. Table Filtering:  Remove Archaea, then by samples and freq
    filtered_table = denoise.table
    if taxonomy_qza:
        try:
            print("[INFO] Filtering out archaea from feature table...")
            tax_art = qiime2.Artifact.load(taxonomy_qza)
            filtered_table = taxa.methods.filter_table(
                table=filtered_table, taxonomy=tax_art, exclude="archaea"
            ).filtered_table
        except Exception as e:
            print(f"[WARN] Could not filter archaea:  {e}")
            # keep original filtered_table

    try:
        print(f"[INFO] Filtering features present in <{min_samples} samples...")
        filtered_table = feature_table.methods.filter_features(
            table=filtered_table, min_samples=min_samples
        ).filtered_table

        print(f"[INFO] Filtering features present with freq ≤{min_frequency} across all samples...")
        filtered_table = feature_table.methods.filter_features(
            table=filtered_table, min_frequency=min_frequency
        ).filtered_table
    except Exception as e:
        print(f"[WARN] Feature filtering failed: {e}")

    # Save filtered table in 04_filtering
    filtering_dir = dirs["filtering"]
    filtered_table_qza = os.path.abspath(os.path.join(filtering_dir, 'table_filtered.qza'))
    filtered_table.save(filtered_table_qza)
    print(f"[INFO] Filtered table saved to {filtered_table_qza}")

    # 5. Feature-table summary (creates table_filtered. qzv)
    if md:
        try:
            table_summary = feature_table.visualizers.summarize(filtered_table, sample_metadata=md)
            table_qzv = os.path.join(filtering_dir, "table_filtered.qzv")
            table_summary.visualization. save(table_qzv)
            print(f"[INFO] table_filtered.qzv created at: {table_qzv}")
        except Exception as e:  
            print(f"[WARN] Could not create table summary visualization: {e}")
    else:
        print("[WARN] Metadata not provided; skipping table summary visualization.")

    # 6. Relative frequency conversion (on filtered table)
    print("[INFO] Calculating relative frequency table (rel-table.qza)...")
    try:
        rel_freq = feature_table.methods.relative_frequency(filtered_table)
        rel_dir = dirs["rel_freq"]
        rel_table_qza = os.path.abspath(os.path.join(rel_dir, 'rel-table.qza'))
        rel_freq.relative_frequency_table.save(rel_table_qza)
        print(f"[INFO] Relative frequency table saved to:  {rel_table_qza}")
        # Try to summarize relative freq table
        try:
            if md:
                rel_summary = feature_table.visualizers.summarize(rel_freq. relative_frequency_table, sample_metadata=md)
                rel_qzv = os.path. join(rel_dir, 'rel-table.qzv')
                rel_summary.visualization.save(rel_qzv)
                print(f"[INFO] rel-table.qzv created at: {rel_qzv}")
        except Exception as e:
            print(f"[WARN] Could not summarize relative frequency table: {e}")
    except Exception as e:  
        print(f"[WARN] Could not compute relative frequency table: {e}")

    # 7. Tree construction (on filtered repseqs)
    print("[INFO] Building phylogenetic tree...")
    try:
        aligned = alignment.methods.mafft(sequences=denoise.representative_sequences)
        masked = alignment.methods.mask(aligned. alignment)
        tree = phylogeny.methods.fasttree(masked. masked_alignment)
        rooted = phylogeny.methods. midpoint_root(tree. tree)
        tree_dir = dirs["tree"]
        rooted_tree_qza = os.path. abspath(os.path.join(tree_dir, 'rooted-tree.qza'))
        rooted.rooted_tree.save(rooted_tree_qza)
        print(f"[INFO] Rooted tree saved to: {rooted_tree_qza}")
    except Exception as e:
        print(f"[WARN] Tree construction failed: {e}")
        rooted = None

    # 8. Custom alpha/beta diversity metrics
    alpha_metrics = []
    if alpha_metrics_arg:
        alpha_metrics = [m.strip() for m in alpha_metrics_arg.split(",") if m.strip()]
    beta_metrics = []
    if beta_metrics_arg:
        beta_metrics = [m.strip() for m in beta_metrics_arg.split(",") if m.strip()]

    if not md:
        print("[INFO] No metadata.  Diversity metrics requiring sample metadata may be skipped where necessary.")

    diversity_dir = dirs["diversity"]
    if depth is not None:
        if alpha_metrics:
            print(f"[INFO] Calculating alpha diversity metrics:  {alpha_metrics}")
            for metric in alpha_metrics:
                try:
                    # Faith PD requires phylogeny
                    if metric == "faith_pd":
                        if rooted is None:
                            raise RuntimeError("Phylogeny not available for faith_pd")
                        res = diversity.pipelines.alpha_phylogenetic(
                            table=filtered_table, phylogeny=rooted.rooted_tree, metric=metric
                        )
                    else:
                        res = diversity.pipelines.alpha(
                            table=filtered_table, metric=metric
                        )
                    outname = os.path.abspath(os.path.join(diversity_dir, f"alpha_{metric}.qza"))
                    res.alpha_diversity.save(outname)
                    print(f"[INFO] Saved {outname}")
                    # Try to run alpha-group-significance visualization if metadata present
                    if md:
                        try:
                            ags = diversity.visualizers.alpha_group_significance(alpha=res.alpha_diversity, metadata=md)
                            ags_qzv = os.path. join(diversity_dir, f"alpha_{metric}_group_significance.qzv")
                            ags.visualization.save(ags_qzv)
                            print(f"[INFO] Saved {ags_qzv}")
                        except Exception as e:  
                            print(f"[WARN] Could not create alpha_group_significance for {metric}: {e}")
                except Exception as ee:
                    print(f"[WARN] Could not calculate alpha {metric}: {ee}")

        if beta_metrics:
            print(f"[INFO] Calculating beta diversity metrics: {beta_metrics}")
            for metric in beta_metrics:
                try:  
                    # Unifrac metrics require phylogeny
                    if metric in ("unweighted_unifrac", "weighted_unifrac"):
                        if rooted is None:
                            raise RuntimeError("Phylogeny not available for UniFrac metrics")
                        res = diversity. pipelines.beta_phylogenetic(
                            table=filtered_table, phylogeny=rooted.rooted_tree, metric=metric
                        )
                    else:
                        res = diversity.pipelines.beta(
                            table=filtered_table, metric=metric
                        )
                    outname = os.path. abspath(os.path.join(diversity_dir, f"beta_{metric}.qza"))
                    # Save distance matrix artifact
                    res.distance_matrix.save(outname)
                    print(f"[INFO] Saved {outname}")
                    # NOTE: PCoA/Emperor visualizations removed due to compatibility issues with some qiime2 versions.  
                    # If you want PCoA/Emperor visualizations, generate them manually in an environment where
                    # diversity. visualizers.pcoa (or appropriate visualizers) is available.
                except Exception as ee:
                    print(f"[WARN] Could not calculate beta {metric}: {ee}")
        if not alpha_metrics and not beta_metrics:
            print("[INFO] No custom diversity metrics specified; skipping alpha/beta diversity metrics.")

        # Optionally run PERMANOVA for one or more metadata columns and for available beta metrics
        if md and permanova and permanova_column:
            # build list of columns (comma-separated allowed)
            cols = [c.strip() for c in permanova_column.split(',') if c.strip()]
            if not cols:
                print("[WARN] No valid permanova_column provided.")
            else:
                # Choose which beta metrics to test PERMANOVA on:  
                # prefer unweighted_unifrac if present, otherwise test all requested beta_metrics
                metrics_to_test = []
                if 'unweighted_unifrac' in beta_metrics:
                    metrics_to_test = ['unweighted_unifrac']
                elif beta_metrics:
                    metrics_to_test = beta_metrics
                else:
                    print("[WARN] No beta metrics available for PERMANOVA; skipping.")
                    metrics_to_test = []

                for metric in metrics_to_test:  
                    dm_path = os.path.abspath(os.path.join(diversity_dir, f"beta_{metric}. qza"))
                    if not os.path.exists(dm_path):
                        print(f"[WARN] Distance matrix not found for metric {metric}:  {dm_path}")
                        continue
                    print(f"[INFO] Running PERMANOVA on metric '{metric}' for columns: {', '.join(cols)}")
                    try:
                        dm_art = qiime2.Artifact.load(dm_path)
                    except Exception as e:
                        print(f"[WARN] Could not load distance matrix {dm_path}: {e}")
                        continue

                    for col in cols:
                        try:
                            metadata_col = md.get_column(col)
                        except Exception as e_col:
                            print(f"[WARN] Could not find metadata column '{col}': {e_col}")
                            continue

                        try:
                            perm_viz = diversity.visualizers.beta_group_significance(
                                distance_matrix=dm_art, metadata=metadata_col, method="permanova", pairwise=True
                            )
                            permanova_qzv = os. path.join(diversity_dir, f"permanova-{metric}-{col}.qzv")
                            perm_viz.visualization.save(permanova_qzv)
                            print(f"[INFO] PERMANOVA results saved:  {permanova_qzv}")
                        except Exception as e_perm:
                            print(f"[WARN] PERMANOVA failed for metric '{metric}', column '{col}': {e_perm}")
        else:
            if permanova and not md:
                print("[WARN] PERMANOVA requested but metadata is missing; skipping PERMANOVA.")
    else:
        print("[WARN] No depth specified for diversity.  Diversity metrics skipped.")

    print("[DONE] Pipeline finished.")
    print(f"[INFO] All results are under: {os.path.join(root, 'results')}")
    print("[INFO] Visualize . qzv files at https://view.qiime2.org")

##########################
# IV. Convert QZA to QZV
##########################

def find_qza_files(directory, pattern="*.qza", recursive=False):
    """
    Find all .qza files in a directory. 
    
    Args:
        directory: Directory to search
        pattern:  Glob pattern for files to match
        recursive: If True, search recursively through subdirectories
    
    Returns:
        List of absolute paths to . qza files
    """
    qza_files = []
    
    if recursive:
        # Walk through all subdirectories
        for root, dirs, files in os. walk(directory):
            for file in files:
                if file. endswith('.qza'):
                    qza_files.append(os.path.join(root, file))
    else:
        # Only search in the specified directory
        qza_files = glob.glob(os.path.join(directory, pattern))
    
    return sorted(qza_files)

def convert_qza_to_qzv(input_path=None, input_dir=None, metadata=None, output=None, pattern="*.qza", recursive=False):
    """
    Convert QIIME2 artifacts (. qza) to visualizations (.qzv).
    
    Supports different artifact types:  
    - FeatureTable[Frequency/RelativeFrequency]:  creates feature-table summary
    - SampleData[AlphaDiversity]: creates alpha-group-significance (requires metadata)
    - DistanceMatrix: creates distance-matrix visualization
    - Phylogeny[Rooted]: creates phylogeny visualization
    - FeatureData[Sequence]: creates tabulate-seqs visualization
    - FeatureData[Taxonomy]: creates metadata tabulation
    
    Args:
        input_path: Path to a single .qza file
        input_dir: Directory containing .qza files
        metadata: Optional metadata file for visualizations that require it
        output: Optional output path for the . qzv file (used only with single input)
        pattern: Glob pattern for files to process when using input_dir (default: "*.qza")
        recursive: If True, search recursively through subdirectories
    """
    # Load metadata if provided
    md = None
    if metadata:  
        if not os.path.exists(metadata):
            print(f"[WARN] Metadata file not found: {metadata}. Metadata-dependent visualizations will be skipped.")
        else:
            try:
                md = qiime2.Metadata. load(metadata)
                print(f"[INFO] Metadata loaded from: {metadata}")
            except Exception as e:
                print(f"[WARN] Could not load metadata {metadata}: {e}")

    # Determine which files to process
    files_to_process = []
    if input_path:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        files_to_process = [input_path]
    elif input_dir:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        files_to_process = find_qza_files(input_dir, pattern=pattern, recursive=recursive)
        if not files_to_process:  
            print(f"[WARN] No . qza files found matching pattern '{pattern}' in {input_dir}")
            if not recursive:
                print("[INFO] Use --recursive to search in subdirectories")
            return
        print(f"[INFO] Found {len(files_to_process)} .qza file(s) to process")
    else:
        raise ValueError("Either --input or --input_dir must be provided")

    # Process each file
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for qza_path in files_to_process: 
        try:
            print(f"\n[INFO] Processing: {qza_path}")
            artifact = qiime2.Artifact.load(qza_path)
            artifact_type = str(artifact.type)
            
            # Determine output path
            if output and len(files_to_process) == 1:
                qzv_path = output
            else:
                qzv_path = qza_path.replace('. qza', '.qzv')
            
            # Check if output already exists
            if os.path. exists(qzv_path):
                print(f"[INFO] Visualization already exists:  {qzv_path}")
                print("[INFO] Skipping (delete existing . qzv to regenerate)")
                skip_count += 1
                continue
            
            print(f"[INFO] Artifact type: {artifact_type}")
            
            # Handle different artifact types
            viz = None
            
            # Feature Tables (frequency or relative frequency)
            if 'FeatureTable[Frequency]' in artifact_type or 'FeatureTable[RelativeFrequency]' in artifact_type:
                if md:
                    viz = feature_table.visualizers. summarize(artifact, sample_metadata=md)
                else:  
                    viz = feature_table.visualizers.summarize(artifact)
                    
            # Alpha Diversity
            elif 'SampleData[AlphaDiversity]' in artifact_type:
                if md:
                    viz = diversity.visualizers.alpha_group_significance(alpha=artifact, metadata=md)
                else:  
                    print("[WARN] Alpha diversity visualization requires metadata.  Skipping.")
                    skip_count += 1
                    continue
                    
            # Distance Matrix
            elif 'DistanceMatrix' in artifact_type:  
                # Basic distance matrix visualization (no metadata required but enhanced with it)
                if md:
                    # Try to create a simple heatmap visualization
                    try:
                        viz = q2_metadata.visualizers.tabulate(artifact. view(qiime2.Metadata))
                    except:  
                        print("[WARN] Could not create distance matrix visualization with metadata")
                        print("[INFO] Use PERMANOVA or PCoA for more detailed analysis")
                        skip_count += 1
                        continue
                else:
                    print("[INFO] Distance matrix loaded.  Use with PCoA/PERMANOVA for visualization.")
                    print("[INFO] Example: qiime diversity pcoa --i-distance-matrix input.qza --o-pcoa output.qza")
                    skip_count += 1
                    continue
                    
            # Phylogenetic Tree
            elif 'Phylogeny[Rooted]' in artifact_type or 'Phylogeny[Unrooted]' in artifact_type:
                print("[INFO] Phylogenetic tree artifact.  Trees are best viewed with external tools.")
                print("[INFO] Export with:  qiime tools export --input-path input.qza --output-path tree/")
                skip_count += 1
                continue
                
            # Feature Data - Sequences
            elif 'FeatureData[Sequence]' in artifact_type:
                viz = feature_table.visualizers.tabulate_seqs(sequences=artifact)
                
            # Feature Data - Taxonomy
            elif 'FeatureData[Taxonomy]' in artifact_type:  
                viz = q2_metadata.visualizers.tabulate(artifact.view(qiime2.Metadata))
                
            else:
                print(f"[WARN] Unsupported artifact type: {artifact_type}")
                print("[INFO] Cannot automatically create visualization for this type.")
                skip_count += 1
                continue
            
            # Save visualization
            if viz:
                viz.visualization.save(qzv_path)
                print(f"[SUCCESS] Visualization saved to: {qzv_path}")
                success_count += 1
                
        except Exception as e:  
            print(f"[ERROR] Failed to process {qza_path}:  {e}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "="*80)
    print("[DONE] Conversion complete.")
    print(f"[SUMMARY] Successfully converted:  {success_count}")
    print(f"[SUMMARY] Skipped: {skip_count}")
    print(f"[SUMMARY] Errors:  {error_count}")
    print(f"[INFO] View .qzv files at https://view.qiime2.org")
    print("="*80)

##########################
# CLI
##########################

def main():
    parser = argparse.ArgumentParser(
        description="QIIME2 16S pipeline: manifest -> import+QC -> denoise+filter+custom diversity -> convert artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Manifest
    p_manifest = subparsers. add_parser("make-manifest", help="Generate QIIME2 manifest from FASTQ folder")
    p_manifest.add_argument('--fastq_dir', type=str, required=True)
    p_manifest.add_argument('--manifest_type', choices=['paired', 'forward', 'reverse'], default='paired')
    p_manifest.add_argument('--manifest_delimiter', choices=[',', '\t'], default='\t')
    p_manifest.add_argument('--manifest_output', type=str, default='manifest.csv')
    p_manifest.add_argument('--read_pattern', choices=['default', 'R'], default='default')

    # 2. Import/QC
    p_import = subparsers.add_parser("import-qc", help="Import FASTQ and generate demux_summary.qzv")
    p_import.add_argument('--manifest', type=str, required=True)
    p_import.add_argument('--paired', action='store_true')
    p_import.add_argument('--demux_out', type=str, default='demux. qza')
    p_import.add_argument('--manifest_delimiter', choices=[',', '\t'], default='\t')

    # 3. Run (denoise + rest)
    p_run = subparsers.add_parser("run", help="Run denoising, taxonomy, filtering, tree, custom diversity")
    p_run.add_argument('--demux', type=str, required=True)
    p_run.add_argument('--classifier', type=str)
    p_run.add_argument('--metadata', type=str)
    p_run.add_argument('--paired', action='store_true')
    p_run.add_argument('--trim_left_f', type=int, default=0)
    p_run.add_argument('--trim_left_r', type=int, default=0)
    p_run.add_argument('--trunc_len_f', type=int, default=240)
    p_run.add_argument('--trunc_len_r', type=int, default=200)
    p_run.add_argument('--depth', type=int, default=10000, help='Sampling depth for diversity steps')
    p_run.add_argument('--permanova', action='store_true')
    p_run.add_argument('--permanova_column', type=str,
                       help="Comma-separated metadata column name(s) to use for PERMANOVA (e.g.  'Treatment,Timepoint')")
    p_run.add_argument('--min_samples', type=int, default=2, help='Minimum #samples for feature retention')
    p_run.add_argument('--min_frequency', type=int, default=10, help='Minimum total frequency for feature retention')
    p_run.add_argument('--alpha_metrics', type=str,
                       help="Comma-separated list of alpha diversity metrics (e.g. shannon,faith_pd,observed_features)")
    p_run.add_argument('--beta_metrics', type=str,
                       help="Comma-separated list of beta diversity metrics (e. g. jaccard,braycurtis,unweighted_unifrac,weighted_unifrac)")

    # 4. Convert QZA to QZV
    p_convert = subparsers.add_parser("convert-to-qzv", help="Convert QIIME2 artifacts (. qza) to visualizations (.qzv)")
    p_convert.add_argument('--input', type=str, help='Path to a single .qza file to convert')
    p_convert.add_argument('--input_dir', type=str, help='Directory containing .qza files to convert')
    p_convert.add_argument('--metadata', type=str, help='Optional metadata file for visualizations that require it')
    p_convert.add_argument('--output', type=str, help='Output path for . qzv file (only used with --input)')
    p_convert.add_argument('--pattern', type=str, default='*.qza', 
                          help='Glob pattern for files to process when using --input_dir (default: *.qza)')
    p_convert.add_argument('--recursive', action='store_true',
                          help='Search recursively through subdirectories (converts ALL . qza files)')

    args = parser.parse_args()
    try:
        if args.command == "make-manifest":
            create_manifest(
                os.path.abspath(args.fastq_dir), args.manifest_output, args.manifest_type,
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
                trunc_len_f=args. trunc_len_f,
                trunc_len_r=args.trunc_len_r,
                depth=args.depth,
                permanova=args. permanova,
                permanova_column=args.permanova_column,
                min_samples=args.min_samples,
                min_frequency=args.min_frequency,
                alpha_metrics_arg=args.alpha_metrics,
                beta_metrics_arg=args.beta_metrics,
            )
        elif args.command == "convert-to-qzv":
            if not args.input and not args.input_dir:
                parser.error("convert-to-qzv requires either --input or --input_dir")
            convert_qza_to_qzv(
                input_path=args.input,
                input_dir=args.input_dir,
                metadata=args.metadata,
                output=args.output,
                pattern=args.pattern,
                recursive=args.recursive
            )
    except Exception as e:
        print(f"[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()
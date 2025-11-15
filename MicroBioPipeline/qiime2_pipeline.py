import os
import csv
import glob
import re
import argparse
import qiime2
from qiime2.plugins import demux, dada2, feature_classifier, feature_table, alignment, phylogeny, diversity

##########################
# Manifest Preparation
##########################

def create_manifest(folder_path, output_file, manifest_type, delimiter=","):
    pattern = re.compile(r"^(?P<sample>[\w-]+)_.*_(R[12])_.*\.fastq\.gz$")
    samples = {}

    fastq_files = glob.glob(os.path.join(folder_path, "*.fastq.gz"))
    for file in fastq_files:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if not match:
            continue

        sample_id = match.group("sample")
        read_direction = match.group(2)
        direction = "forward" if read_direction == "R1" else "reverse"
        samples.setdefault(sample_id, {})[direction] = os.path.abspath(file)

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
                    print(f"[WARN] sample '{sample_id}' missing forward or reverse read.")
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
    print(f"Manifest file generated: {output_file}")

##########################
# QIIME2 Pipeline
##########################

def qiime2_pipeline(args):
    # Import data
    print("Importing FASTQ data using manifest...")
    if args.paired:
        view_type = 'PairedEndFastqManifestPhred33V2'
        imp_type = 'SampleData[PairedEndSequencesWithQuality]'
    else:
        view_type = 'SingleEndFastqManifestPhred33V2'
        imp_type = 'SampleData[SequencesWithQuality]'
    demuxed = qiime2.Artifact.import_data(
        imp_type, args.manifest, view_type=view_type
    )

    # Summarize demux
    print("Summarizing demultiplexed quality...")
    demux.visualizers.summarize(demuxed).visualization.save('demux_summary.qzv')

    # DADA2
    print("Running DADA2...")
    if args.paired:
        denoise = dada2.methods.denoise_paired(
            demultiplexed_seqs=demuxed,
            trim_left_f=args.trim_left_f,
            trim_left_r=args.trim_left_r,
            trunc_len_f=args.trunc_len_f,
            trunc_len_r=args.trunc_len_r
        )
    else:
        denoise = dada2.methods.denoise_single(
            demultiplexed_seqs=demuxed,
            trim_left=args.trim_left_f,
            trunc_len=args.trunc_len_f
        )
    denoise.table.save('table.qza')
    denoise.representative_sequences.save('rep-seqs.qza')
    denoise.denoising_stats.save('denoising-stats.qza')

    # Taxonomy assignment
    print("Classifying taxonomy...")
    taxonomy = feature_classifier.methods.classify_sklearn(
        reads=denoise.representative_sequences,
        classifier=qiime2.Artifact.load(args.classifier)
    )
    taxonomy.classification.save('taxonomy.qza')

    # Feature table summary
    print("Feature table summarization...")
    feature_table.visualizers.summarize(
        denoise.table,
        qiime2.Metadata.load(args.metadata)
    ).visualization.save('table.qzv')

    # Relative abundance
    print("Calculating relative abundance...")
    rel_freq = feature_table.methods.relative_frequency(denoise.table)
    rel_freq.relative_frequency_table.save('rel-table.qza')

    # Phylogenetic tree
    print("Generating phylogenetic tree...")
    aligned = alignment.methods.mafft(
        sequences=denoise.representative_sequences
    )
    masked = alignment.methods.mask(aligned.alignment)
    tree = phylogeny.methods.fasttree(masked.masked_alignment)
    rooted = phylogeny.methods.midpoint_root(tree.tree)
    rooted.rooted_tree.save('rooted-tree.qza')

    # Diversity metrics
    print("Running diversity (core metrics)...")
    core_metrics = diversity.pipelines.core_metrics_phylogenetic(
        table=denoise.table,
        phylogeny=rooted.rooted_tree,
        metadata=qiime2.Metadata.load(args.metadata),
        sampling_depth=args.depth
    )
    core_metrics.faith_pd_vector.save('faith_pd_vector.qza')
    core_metrics.unweighted_unifrac_distance_matrix.save('unweighted_unifrac.qza')
    core_metrics.unweighted_unifrac_pcoa_results.save('unweighted_unifrac_pcoa.qza')
    core_metrics.beta_group_significance.save('beta-group-significance.qzv')

    # PERMANOVA
    if args.permanova and args.permanova_column:
        print(f"Performing PERMANOVA on {args.permanova_column}...")
        permanova = diversity.visualizers.beta_group_significance(
            distance_matrix=core_metrics.unweighted_unifrac_distance_matrix,
            metadata=qiime2.Metadata.load(args.metadata).get_column(args.permanova_column),
            method="permanova",
            pairwise=True
        )
        permanova.visualization.save(f"permanova-{args.permanova_column}.qzv")
        print(f"PERMANOVA result: permanova-{args.permanova_column}.qzv")

    print("Pipeline finished. Visualize .qzv files at https://view.qiime2.org")

##########################
# ARGPARSE SETUP
##########################

def main():
    parser = argparse.ArgumentParser(
        description="QIIME2 16S rRNA pipeline (manifest, analysis, taxonomy, tree, diversity, PERMANOVA)"
    )

    # Manifest creation options
    parser.add_argument('--make_manifest', action='store_true',
                        help='Generate QIIME2 manifest file from FASTQ folder')
    parser.add_argument('--fastq_dir', type=str,
                        help='Directory of FASTQ files for manifest')
    parser.add_argument('--manifest_type', choices=['paired', 'forward', 'reverse'],
                        default='paired', help='Manifest type')
    parser.add_argument('--manifest_delimiter', choices=[',', '\t'], default=',',
                        help='Delimiter for manifest file (default: csv)')

    parser.add_argument('--manifest_output', type=str, default='manifest.csv',
                        help='Output manifest filename (default: manifest.csv)')

    # Pipeline options
    parser.add_argument('--manifest', type=str, required=False,
                        help='Manifest file (csv or tsv) for pipeline input')
    parser.add_argument('--classifier', type=str, help='QIIME2 .qza classifier for taxonomy')
    parser.add_argument('--metadata', type=str, help='QIIME2 sample metadata .tsv')
    parser.add_argument('--paired', action='store_true', help='Set for paired-end reads')
    parser.add_argument('--trim_left_f', type=int, default=0, help='Trim left bases (forward)')
    parser.add_argument('--trim_left_r', type=int, default=0, help='Trim left bases (reverse)')
    parser.add_argument('--trunc_len_f', type=int, default=240, help='Truncate length (forward)')
    parser.add_argument('--trunc_len_r', type=int, default=200, help='Truncate length (reverse)')
    parser.add_argument('--depth', type=int, default=10000, help='Sampling depth for diversity')
    parser.add_argument('--permanova', action='store_true', help='Perform PERMANOVA test')
    parser.add_argument('--permanova_column', type=str, help='Metadata column for PERMANOVA')

    args = parser.parse_args()

    # Manifest generation
    if args.make_manifest:
        if not args.fastq_dir:
            print("[ERROR] --fastq_dir required when using --make_manifest")
            exit(1)
        manifest_path = args.manifest_output
        create_manifest(os.path.abspath(args.fastq_dir), manifest_path, args.manifest_type,
                        delimiter=args.manifest_delimiter)
        print(f"[INFO] Manifest preparation finished: {manifest_path}")
        exit(0)

    # Run QIIME2 pipeline only if manifest is provided
    if args.manifest:
        qiime2_pipeline(args)
    else:
        print("[ERROR] --manifest <manifest_file> required to run QIIME2 pipeline.")


if __name__ == "__main__":
    main()

    # python qiime2_full_pipeline.py \
    #     --make_manifest \
    #     --fastq_dir /home/you/fastq_data \
    #     --manifest_type forward \                                  -> (paired, forward, reverse)
    #     --manifest_output manifest.tsv \
    #     --manifest_delimiter "\t"                                  -> ("," or "\t") if csv facultative

    # python qiime2_full_pipeline.py \
    #     --manifest manifest.csv \                                  -> generated manifest file .csv or .tsv
    #     --classifier /home/you/silva-138-99-nb-classifier.qza \
    #     --metadata /home/you/sample-metadata.tsv \
    #     --paired \                                                 -> (paired, forward, reverse)
    #     --trim_left_f 0 --trim_left_r 0 \
    #     --trunc_len_f 240 --trunc_len_r 200 \
    #     --depth 10000 \
    #     --permanova \
    #     --permanova_column Treatment
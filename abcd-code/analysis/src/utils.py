from src.SynthSeg import SynthSeg
from src.StatisticalAnalysis import StatisticalAnalysis


def get_sig_rois(labels_df, run_qc=True):
    synthseg = SynthSeg(
        subjects=labels_df.index,
        run_qc=run_qc,
        run_qc_min_score=0.65,
        run_qc_max_failed_regions=1,
    )
    if run_qc:
        synthseg_df = synthseg.qc_df
    else:
        synthseg_df = synthseg.df

    synthseg_with_labels = synthseg_df.join(labels_df)

    control_df = synthseg_df[synthseg_with_labels[labels_df.name] == 0]
    disorder_df = synthseg_df[synthseg_with_labels[labels_df.name] == 1]

    print(labels_df.name.split("_")[0].upper())
    print("-" * 10)
    print("# control: \t\t", control_df.shape[0])
    print("# disorder: \t\t", disorder_df.shape[0])

    synthseg_conditions = StatisticalAnalysis(
        control_df,
        disorder_df,
        ind=True,
        scale=False,
        dataset_names=["control", "disorder"],
    )

    synthseg_sig_vols, synthseg_all_stats = synthseg_conditions.compare()

    print(f"# different regions: \t {synthseg_sig_vols.shape[0]}")

    return synthseg_sig_vols, synthseg_conditions

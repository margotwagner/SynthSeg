import pandas as pd


class SynthSeg:
    def __init__(
        self,
        subjects="all",
        run_qc=False,
        run_qc_min_score=0.65,
        run_qc_max_failed_regions=1
    ):
        self.dir = "/cnl/abcd/data/imaging/smri/interim/synthseg/baseline/"
        self.subjects = subjects
        self.df = self.df()
        self.qc_scores = self.qc_scores()
        if run_qc:
            self.run_qc_min_score = run_qc_min_score
            self.run_qc_max_failed_regions = run_qc_max_failed_regions
            self.qc_df = self.qc_df(self.run_qc_min_score, self.run_qc_max_failed_regions)

    def df(self):
        """Attribute setter for SynthSeg ROI volumes"""
        # Subject volumes
        vol_df = pd.read_csv(self.dir + "vol.csv", sep=",")
        vol_df["subject"] = vol_df["subject"].str[4:19]  # subject GUIDs
        vol_df = vol_df.set_index("subject")

        # Take subset of subjects if specified

        if type(self.subjects) != str:
            vol_df = vol_df[vol_df.index.isin(self.subjects)]

        return vol_df

    def qc_scores(self):
        """Attribute setter for qc scores"""
        # Subject QC scores
        df = pd.read_csv(self.dir + "qc.csv", sep=",")
        df["subject"] = df["subject"].str[4:19]  # subject GUIDS
        df = df.set_index("subject")

        # Take subset of subjects if specified
        if type(self.subjects) != str:
            df = df[df.index.isin(self.subjects)]

        return df

    def get_subjects(self):
        return list(self.df.index.values)

    def get_num_subjects(self):
        return len(self.get_subjects())

    def qc_regions_to_exclude(
        self,
        min_score=0.65,
        max_failed_regions=1,
    ):
        """Subjects to exclude based on defined exclusion criteria.

        Default values for exclusion criteria based on Billot et al SynthSeg+ paper.
        """
        # Inclusion criteria based on automatic SynthSeg QC score
        failed_qc_regions = pd.DataFrame(
            {
                "subject": pd.Series(dtype="str"),
                "n_fails": pd.Series(dtype="int"),
                "region": pd.Series(dtype="str"),
                "qc_score": pd.Series(dtype="float"),
            }
        )
        for index, row in self.qc_scores.iterrows():
            n_fails = (row < min_score).sum()

            if n_fails >= max_failed_regions:
                failed_row = row.loc[row < min_score]

                for i in range(n_fails):
                    entry = {
                        "subject": index,
                        "n_fails": n_fails,
                        "region": failed_row.index[i],
                        "qc_score": failed_row[i],
                    }
                    failed_qc_regions = pd.concat(
                        [failed_qc_regions, pd.DataFrame([entry])],
                        axis=0,
                        ignore_index=True,
                    )

        return failed_qc_regions

    def qc_subjects_to_exclude(
        self,
        min_score=0.65,
        max_failed_regions=1,
    ):
        """Subjects that did not pass qc check."""
        failed_qc_regions = self.qc_regions_to_exclude(
            min_score=min_score, max_failed_regions=max_failed_regions
        )

        return list(failed_qc_regions["subject"].unique())

    def qc_df(
        self,
        min_score=0.65,
        max_failed_regions=1,
    ):
        """Synthseg output df with only subjects that pass qc check."""
        subjects_to_exclude = self.qc_subjects_to_exclude(
            min_score=min_score, max_failed_regions=max_failed_regions
        )

        qc_df = self.df.copy()

        qc_df = qc_df.drop(subjects_to_exclude, axis=0)

        return qc_df

    def qc_subjects_to_include(
        self,
        min_score=0.65,
        max_failed_regions=1,
    ):
        qc_df = self.qc_df(min_score=min_score, max_failed_regions=max_failed_regions)

        return list(qc_df.index.values)

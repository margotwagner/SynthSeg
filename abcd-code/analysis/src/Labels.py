import pandas as pd


class Labels:
    def __init__(self):
        self.dir = "/cnl/abcd/data/tabular/raw/"
        self.save_dir = "/cnl/abcd/data/labels/"
        self.shortname = "abcd_cbcls01"
        self.cbcl_df = self.cbcl_df()
        self.dsm_df = self.dsm_df()
        self.aseba_df = self.aseba_df()
        self.ctrl_subj = self.ctrl_subj()
        self.depr_dsm_df = self.depr_dsm_df()
        self.anx_dsm_df = self.anx_dsm_df()
        self.adhd_dsm_df = self.adhd_dsm_df()
        self.opposit_dsm_df = self.opposit_dsm_df()
        self.conduct_dsm_df = self.conduct_dsm_df()
        self.somatic_dsm_df = self.somatic_dsm_df()

    def cbcl_df(self):
        """
        Preprocesses the dataframe by performing the following steps:
        1. Reads in a txt file, skipping descriptions.
        2. Filters the dataframe to only include baseline data.
        3. Extracts all features from the baseline data.
        4. Filters out irrelevant columns.
        5. Drops rows with NaN values.
        6. Renames the labels to be more human-readable.
        7. Modifies the index of the dataframe.

        Returns:
            df (pandas.DataFrame): The preprocessed dataframe.
        """

        # read in txt file, skipping descriptions
        raw_inst_df = pd.read_csv(
            "{}{}.txt".format(self.dir, self.shortname), sep="\t", low_memory=False
        ).iloc[1:, :]

        # just take baseline data for now
        baseline = raw_inst_df[raw_inst_df["eventname"] == "baseline_year_1_arm_1"]

        # get all feats
        feats = baseline.keys().tolist()[9:-2]

        # ids
        ids = [
            "src_subject_id",
            "interview_date",
            "interview_age",
            "sex",
        ]

        # isolate t-scores
        feats = [f for f in feats if f.split("_")[-1] == "t"]

        # filter out remaining columns
        baseline = baseline.filter(ids + feats, axis=1).set_index("src_subject_id")

        # Drop NaN rows
        baseline.dropna(inplace=True)

        # filter to only include columns of interest
        df = baseline.filter(feats, axis=1).astype("float")

        # rename labels to be more human-readable
        labels = [k.split("_")[3] + "_" + k.split("_")[2] for k in df.keys()]

        rename_dict = {}
        for old, new in zip(df.keys(), labels):
            rename_dict[old] = new

        df.rename(columns=rename_dict, inplace=True)

        df.index = [i.replace("_", "") for i in df.index]

        return df

    def aseba_df(self):
        """
        Returns a DataFrame containing only the ASEBA syndrome classifications from the CBCL DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing only the DSM classifications.
        """
        # just looking at syndrome classifications
        aseba_feats = [k for k in self.cbcl_df.keys() if "syn" in k]

        aseba_df = self.cbcl_df.filter(aseba_feats, axis=1)

        return aseba_df

    def dsm_df(self):
        """
        Returns a DataFrame containing only the DSM classifications from the cbcl_df DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing only the DSM classifications.
        """
        # just looking at dsm classifications
        dsm_feats = [k for k in self.cbcl_df.keys() if "dsm" in k]

        dsm_df = self.cbcl_df.filter(dsm_feats, axis=1)

        return dsm_df

    def ctrl_subj(self):
        """
        Returns a list of control subjects defined as measuring below the threshold.

        Returns:
            list: A list of control subjects.
        """
        ctrl_subj = self.dsm_df.loc[
            (
                (self.dsm_df["depress_dsm5"] == 50.0)
                & (self.dsm_df["anxdisord_dsm5"] == 50.0)
                & (self.dsm_df["somaticpr_dsm5"] == 50.0)
                & (self.dsm_df["adhd_dsm5"] == 50.0)
                & (self.dsm_df["opposit_dsm5"] == 50.0)
                & (self.dsm_df["conduct_dsm5"] == 50.0)
            )
        ].index

        return list(ctrl_subj)

    def build_disorder_df(self, disorder):
        """
        Returns a DataFrame containing the disorder labels for disordered subjects.

        Returns:
            pandas.DataFrame: A DataFrame with disorder labels for disordered subjects.
        """

        # clinically depressed subjects
        disorder_subj = self.dsm_df[self.dsm_df[disorder] > 69.0].index

        subj = self.ctrl_subj + list(disorder_subj)

        df = self.dsm_df[disorder].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df

    def depr_dsm_df(self):
        """
        Returns a DataFrame containing the depression labels for clinically depressed subjects.

        Returns:
            pandas.DataFrame: A DataFrame with depression labels for clinically depressed subjects.
        """

        return self.build_disorder_df("depress_dsm5")

    def anx_dsm_df(self):
        """
        Returns a DataFrame containing structural volumes for anxiety disorders based on DSM-5 criteria.
        """

        return self.build_disorder_df("anxdisord_dsm5")

    def adhd_dsm_df(self):
        """
        Returns a dataframe containing structural volumes for ADHD (Attention-Deficit/Hyperactivity Disorder) based on DSM-5 criteria.
        """
        return self.build_disorder_df("adhd_dsm5")

    def opposit_dsm_df(self):
        """
        Returns a DataFrame containing containing structural volumes for oppositional defiance disorder labels based on DSM-5 criteria.
        """
        return self.build_disorder_df("opposit_dsm5")

    def conduct_dsm_df(self):
        """
        Returns a DataFrame containing structural volumes for conduct disorder labels based on DSM-5 criteria.
        """
        return self.build_disorder_df("conduct_dsm5")

    def somatic_dsm_df(self):
        """
        Returns a DataFrame containing structural volumes for somatic disorder labels based on DSM-5 criteria.
        """

        return self.build_disorder_df("somaticpr_dsm5")

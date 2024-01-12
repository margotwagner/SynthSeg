import pandas as pd


class Labels:
    def __init__(self, instrument="cbcl", disorder_type="dsm"):
        self.dir = "/cnl/abcd/data/tabular/raw/"
        self.save_dir = "/cnl/abcd/data/labels/"
        self.instrument = instrument
        self.disorder_type = disorder_type
        if self.instrument == "cbcl":
            self.shortname = "abcd_cbcls01"
            self.cbcl_df = self.cbcl_df()
            if self.disorder_type == "dsm":
                self.dsm_df = self.dsm_df()
                self.ctrl_subj = self.ctrl_subj()
                self.depr_dsm_df = self.depr_dsm_df()
                self.anx_dsm_df = self.anx_dsm_df()
                self.adhd_dsm_df = self.adhd_dsm_df()
                self.opposit_dsm_df = self.opposit_dsm_df()
                self.conduct_dsm_df = self.conduct_dsm_df()
                self.somatic_dsm_df = self.somatic_dsm_df()
            elif self.disorder_type == "aseba":
                self.aseba_df = self.aseba_df()
                self.ctrl_subj = self.ctrl_subj()
                self.attent_aseba_df = self.attent_aseba_df()
                self.aggro_aseba_df = self.aggro_aseba_df()
                self.anxdep_aseba_df = self.anxdep_aseba_df()
                self.withdep_aseba_df = self.withdep_aseba_df()
                self.somatic_aseba_df = self.somatic_aseba_df()
                self.social_aseba_df = self.social_aseba_df()
                self.thought_aseba_df = self.thought_aseba_df()
                self.rulebreak_aseba_df = self.rulebreak_aseba_df()
        elif self.instrument == "ksad":
            print("ksad functionality not yet implemented")
        else:
            print("Instrument not recognized")
        self.metadata_df = self.metadata_df()

    def get_handedness(self):
        # read in txt file, skipping descriptions
        raw_inst_df = pd.read_csv(
            "{}{}.txt".format(self.dir, "abcd_ehis01"), sep="\t", low_memory=False
        ).iloc[1:, :]

        # just take baseline data for now
        baseline = raw_inst_df[raw_inst_df["eventname"] == "baseline_year_1_arm_1"]

        feats = ["src_subject_id", "ehi1b"]  # writing hand
        # NOTE: "If mixed handed, use the hand that the child writes with for NeuroCog and fMRI tasks." -> not using summary score due to this

        # filter to only include columns of interest & set index
        df = baseline.filter(feats, axis=1).set_index("src_subject_id")

        # Drop NaN rows
        df.dropna(inplace=True)

        # convert to numeric
        df = df.astype("int")

        # map to binary left or right handed
        df["hand"] = (df["ehi1b"] >= 0).astype(int)
        df["hand"].replace({0: "L", 1: "R"}, inplace=True)
        df.drop("ehi1b", axis=1, inplace=True)

        # remove underscores from index
        df.index = [i.replace("_", "") for i in df.index]

        return df

    def metadata_df(self):
        """
        Returns a DataFrame containing the patient metadata for the CBCL dataset.

        Currently includes subject sex and age.
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
        df = baseline.filter(
            [
                "sex",
                "interview_age",
            ],
            axis=1,
        )

        # rename age labels to be more human-readable
        df.rename(columns={"interview_age": "age"}, inplace=True)

        # remove underscores from index
        df.index = [i.replace("_", "") for i in df.index]

        # get handedness
        hand_df = self.get_handedness()

        # merge handedness with metadata
        df = df.merge(hand_df, how="inner", left_index=True, right_index=True)

        return df

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

        # remove underscores from index
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
        # NOTE: temporary fix until there's time to look into why the last 3 columns don't match the same numerical format as the rest
        aseba_feats = aseba_feats[:-3]

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
        if self.disorder_type == "dsm":
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
        elif self.disorder_type == "aseba":
            ctrl_subj = self.aseba_df.loc[
                (
                    (self.aseba_df["anxdep_syn"] == 50.0)
                    & (self.aseba_df["withdep_syn"] == 50.0)
                    & (self.aseba_df["somatic_syn"] == 50.0)
                    & (self.aseba_df["social_syn"] == 50.0)
                    & (self.aseba_df["thought_syn"] == 50.0)
                    & (self.aseba_df["attention_syn"] == 50.0)
                    & (self.aseba_df["rulebreak_syn"] == 50.0)
                    & (self.aseba_df["aggressive_syn"] == 50.0)
                )
            ].index

        return list(ctrl_subj)

    def build_disorder_df(self, disorder):
        """
        Returns a DataFrame containing the disorder labels for disordered subjects.

        Returns:
            pandas.DataFrame: A DataFrame with disorder labels for disordered subjects.
        """
        if self.disorder_type == "dsm":
            disorder_df = self.dsm_df
        elif self.disorder_type == "aseba":
            disorder_df = self.aseba_df

        # clinically depressed subjects
        disorder_subj = disorder_df[disorder_df[disorder] > 69.0].index

        subj = self.ctrl_subj + list(disorder_subj)

        df = disorder_df[disorder].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df

    def depr_dsm_df(self):
        """
        Returns a DataFrame containing the depression labels for clinically depressed subjects.
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

    def anxdep_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Anxious/Depressed subjects.
        """

        return self.build_disorder_df("anxdep_syn")

    def withdep_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Withdrawn/Depressed subjects.
        """

        return self.build_disorder_df("withdep_syn")

    def somatic_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Somatic Complaints subjects.
        """

        return self.build_disorder_df("somatic_syn")

    def social_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clinical ASEBA Social Problem subjects.
        """

        return self.build_disorder_df("social_syn")

    def thought_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Thought Problem subjects.
        """

        return self.build_disorder_df("thought_syn")

    def attent_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Attention Problem subjects.
        """

        return self.build_disorder_df("attention_syn")

    def rulebreak_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Rule Breaking Behavior subjects.
        """

        return self.build_disorder_df("rulebreak_syn")

    def aggro_aseba_df(self):
        """
        Returns a DataFrame containing the depression labels for clincal ASEBA Aggressive Behavior subjects.
        """

        return self.build_disorder_df("aggressive_syn")

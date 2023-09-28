import pandas as pd


class Labels():

    def __init__(self):

        self.dir = '/cnl/abcd/data/tabular/raw/'
        self.save_dir = '/cnl/abcd/data/labels/'
        self.shortname = 'abcd_cbcls01'
        self.dsm_df = self.dsm_df()
        self.depr_df = self.depr_df()
        self.anx_df = self.anx_df()
        self.adhd_df = self.adhd_df()
        self.opposit_df = self.opposit_df()
        self.conduct_df = self.conduct_df()
        self.somaticpr_df = self.somaticpr_df()
        
    def dsm_df(self):
        # read in txt file, skipping descriptions
        raw_inst_df = pd.read_csv(
            "{}{}.txt".format(
                self.dir,
                self.shortname
                ),
            sep='\t',
            low_memory=False
            ).iloc[1:, :]

        # just take baseline data for now
        baseline = raw_inst_df[
            raw_inst_df["eventname"] == 'baseline_year_1_arm_1'
            ]

        # get all feats
        feats = baseline.keys().tolist()[9:-2]

        # ids
        ids = [
            'src_subject_id',
            'interview_date',
            'interview_age',
            'sex',
            ]

        # isolate t-scores
        feats = [f for f in feats if f.split('_')[-1] == 't']

        # filter out remaining columns
        baseline = baseline.filter(
            ids + feats,
            axis=1
        ).set_index('src_subject_id')

        # Drop NaN rows
        baseline.dropna(inplace=True)

        # filter to only include columns of interest
        df = baseline.filter(
            feats,
            axis=1
        ).astype('float')

        # rename labels to be more human-readable
        labels = [
            k.split('_')[3] + '_' + k.split('_')[2] for k in df.keys()
            ]

        rename_dict = {}
        for old, new in zip(df.keys(), labels):
            rename_dict[old] = new

        df.rename(columns=rename_dict, inplace=True)

        df.index = [i.replace("_", "") for i in df.index]

        # just looking at dsm classifications
        dsm_feats = [k for k in df.keys() if 'dsm' in k]

        dsm_df = df.filter(
            dsm_feats,
            axis=1
        )
        
        return dsm_df

    def depr_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["depress_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["depress_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df
    
    def anx_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["anxdisord_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["anxdisord_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df
    
    def adhd_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["adhd_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["adhd_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df
    
    def opposit_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["opposit_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["opposit_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df
    
    def conduct_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["conduct_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["conduct_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df
    
    def somaticpr_df(self):
        
        # healthy subjects
        healthy_subj = self.dsm_df.loc[
            (
                (self.dsm_df['depress_dsm5'] == 50.0)
                & (self.dsm_df['anxdisord_dsm5'] == 50.0)
                & (self.dsm_df['somaticpr_dsm5'] == 50.0)
                & (self.dsm_df['adhd_dsm5'] == 50.0)
                & (self.dsm_df['opposit_dsm5'] == 50.0)
                & (self.dsm_df['conduct_dsm5'] == 50.0)
            )
            ].index

        # clinically depressed subjects
        depress_subj = self.dsm_df[self.dsm_df["somaticpr_dsm5"] > 69.0].index

        subj = list(healthy_subj) + list(depress_subj)

        df = self.dsm_df["somaticpr_dsm5"].loc[subj]

        df = (df > 69.0).astype(int)

        df.index = [i.replace("_", "") for i in df.index]

        return df

    def save_df(self):
        self.depr_df.to_csv(
            self.save_dir + "baseline-bin-healthy-depress.csv",
            sep=","
        )

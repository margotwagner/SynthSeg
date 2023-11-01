import pandas as pd
import json


class ABCD:
    def __init__(
        self,
        subjects="all",
        with_qc=True,
    ):
        self.subjects = subjects  # subset of subjects (optional)
        self.dir = "/cnl/abcd/data/tabular/raw/"
        self.dict_dir = "/cnl/abcd/data/abcd-metadata/abcd-4.0-data-dictionaries/"
        self.save_dir = "/cnl/abcd/data/imaging/smri/interim/synthseg/"
        self.shortname = "abcd_smrip10201"
        self.qc_shortname = "abcd_imgincl01"
        self.mri_info_shortname = "abcd_mri01"
        self.with_qc = with_qc
        self.df = self.df()
        self.dict = self.dict()
        self.qc_scores = self.qc_scores()
        self.qc_df = self.qc_df()
        if with_qc:
            self.df = self.qc_df
        self.mri_info_df = self.mri_info_df()

    def df(self):
        df = pd.read_csv(self.dir + self.shortname + ".txt", sep="\t", dtype="str")[1:]
        df = df[df["eventname"] == "baseline_year_1_arm_1"]
        df["subjectkey"] = [i.replace("_", "") for i in df["subjectkey"]]
        df = df.set_index("subjectkey")

        # Subset if specified
        if type(self.subjects) != str:
            df = df[df.index.isin(self.subjects)]

        # ROI volumes as features
        feats = [k for k in df.keys() if "vol_cdk" in k or "vol_scs" in k]
        df = df[df.columns & feats]

        # Drop lesions (NaN) columns
        df = df.drop(["smri_vol_scs_lesionlh", "smri_vol_scs_lesionrh"], axis=1).astype(
            "float"
        )

        return df

    def mri_info_df(self):
        df = pd.read_csv(
            self.dir + self.mri_info_shortname + ".txt", sep="\t", dtype="str"
        )[1:]
        df = df[df["eventname"] == "baseline_year_1_arm_1"]
        df["subjectkey"] = [i.replace("_", "") for i in df["subjectkey"]]
        df = df.set_index("subjectkey")
        df = df[
            [
                "mri_info_manufacturer",
                "mri_info_manufacturersmn",
                "mri_info_deviceserialnumber",
            ]
        ]

        df = df.rename(
            {
                "mri_info_manufacturer": "manufacturer",
                "mri_info_manufacturersmn": "model",
                "mri_info_deviceserialnumber": "serial_number",
            },
            axis=1,
        )

        df["manufacturer"] = df["manufacturer"].replace(
            {
                "GE MEDICAL SYSTEMS": "GE",
                "Philips Medical Systems": "Philips",
                "SIEMENS": "Siemens",
            }
        )

        # Subset if specified
        if type(self.subjects) != str:
            df = df[df.index.isin(self.subjects)]

        if self.with_qc:
            df = df[df.index.isin(self.qc_df.index)]

        return df

    def qc_scores(self):
        # Recommended Imaging Inclusion
        df = pd.read_csv(self.dir + self.qc_shortname + ".txt", sep="\t", dtype="str")[
            1:
        ]
        df = df[df["eventname"] == "baseline_year_1_arm_1"]
        df["subjectkey"] = [i.replace("_", "") for i in df["subjectkey"]]
        df = df.set_index("subjectkey")

        # Subset if specified
        if type(self.subjects) != str:
            df = df[df.index.isin(self.subjects)]

        df = df["imgincl_t1w_include"]  # 0 = No; 1 = Yes
        df = df.rename("include")

        return df.astype("int")

    def qc_subjects_to_exclude(self):
        return self.qc_scores[self.qc_scores == 0].index

    def qc_subjects_to_include(self):
        return self.qc_scores[self.qc_scores == 1].index

    def dict(self):
        dict = pd.read_csv(self.dict_dir + self.shortname + ".csv")
        return dict

    def nan_check(self):
        nan_columns = []
        for k in self.df.keys():
            if self.df[k].isnull().sum() != 0:
                nan_columns.append(k, self.df[k].isnull().sum())
        return nan_columns

    def get_regions(self):
        regions = []
        for f in self.df.keys():
            if f == "depress_dsm5":
                continue
            desc = self.dict[self.dict["ElementName"] == f][
                "ElementDescription"
            ].values[0]
            if "total" not in f:
                regions.append(desc.split("ROI ")[1])
            else:
                regions.append(f.split("_")[-1])

        return regions

    def get_abcd_to_synthseg_label_dict(self, save=False):
        abcd_to_synthseg = {}

        for abcd, desc in zip(self.df.keys(), self.get_regions()):
            # cdk (cortex parcellation)
            if "cdk" in abcd:
                if "Banks of Superior Temporal Sulcus" in desc:
                    desc = desc.replace("Banks of Superior Temporal Sulcus", "bankssts")
                if "total" in desc:
                    if "rh" in desc:
                        desc = desc.replace("totalrh", "right cerebral cortex")
                    elif "lh" in desc:
                        desc = desc.replace("totallh", "left cerebral cortex")
                    abcd_to_synthseg[abcd] = desc
                else:
                    abcd_to_synthseg[abcd] = "ctx-" + desc
            # scs (subcortical segmentation)
            else:
                # specific changes
                if "ventraldc" in desc:
                    desc = desc.replace("ventraldc", "ventral DC")
                elif "thalamus" in desc:
                    desc = desc.replace("thalamus-proper", "thalamus")
                elif "inf-lat-vent" in desc:
                    desc = desc.replace("inf-lat-vent", "inferior lateral ventricle")
                elif desc == "wholebrain":
                    desc = "total intracranial"

                if desc == "brain-stem":
                    abcd_to_synthseg[abcd] = desc
                else:
                    abcd_to_synthseg[abcd] = desc.replace("-", " ")

        if save:
            with open(self.save_dir + "abcd_to_synthseg_labels.txt", "w+") as f:
                json.dump(abcd_to_synthseg, f)

        return abcd_to_synthseg

    def rename_to_synthseg_labels(self):
        abcd_to_synthseg = self.get_abcd_to_synthseg_label_dict()
        self.df = self.df.rename(columns=abcd_to_synthseg)
        return self.df

    def match_synthseg_regions(self, synthseg_df):
        """Rename ROI labels and drop ROI columns that are not present in SynthSeg."""
        self.df = self.rename_to_synthseg_labels()

        to_drop = set(self.df.columns) - set(synthseg_df.columns)
        self.df = self.df.drop(list(to_drop), axis=1)
        return self.df

    def qc_df(self):
        """ABCD ROI volumes with only subjects that pass qc check."""
        qc_df = self.df.copy()

        # Remove failed subjects
        qc_df = qc_df[qc_df.index.isin(self.qc_scores[self.qc_scores == 1].index)]

        return qc_df

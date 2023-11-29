from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
)
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'


class StatisticalAnalysis:
    def __init__(
        self,
        dataset_1,
        dataset_2,
        ind=True,
        scale=False,  # whether to scale the data
        verbose=False,
        test_normalcy=False,
        scale_type="minmax",
        dataset_names=["dataset_1", "dataset_2"],
    ):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ind = ind
        self.scale = scale
        if self.scale:
            self.scale_data(type=self.scale_type)
        self.verbose = verbose
        self.test_normalcy = test_normalcy
        self.dataset_names = dataset_names
        self.pvalue_threshold = 0.05
        self.scale_type = scale_type

    def scale_data(self, type):
        # min-max scale the data
        if type == "minmax":
            scaler = MinMaxScaler()
        elif type == "std":
            scaler = StandardScaler()
        elif type == "robust":
            scaler = RobustScaler()
        elif type == "gaussian":
            scaler = QuantileTransformer(output_distribution="normal")
        scaled_1 = scaler.fit_transform(self.dataset_1.to_numpy())
        scaled_2 = scaler.fit_transform(self.dataset_2.to_numpy())

        # convert back to dataframe
        self.dataset_1 = pd.DataFrame(
            scaled_1, columns=self.dataset_1.columns, index=self.dataset_1.index
        )
        self.dataset_2 = pd.DataFrame(
            scaled_2, columns=self.dataset_2.columns, index=self.dataset_2.index
        )

    def is_normal(self, feat):
        is_1_normal = False
        is_2_normal = False

        # Check if dataset 1 is normally distributed
        if len(self.dataset_1[feat]) < 5000:
            # For small samples, use shapiro-wilk
            if len(self.dataset_1[feat]) < 50:
                _, p = stats.shapiro(self.dataset_1[feat])
            # For midsize samples, use kolmogorov-smirnov
            else:
                _, p = stats.kstest(self.dataset_1[feat], stats.norm.cdf)
            # If p < 0.05, reject hypothesis that data is normal
            if p >= self.pvalue_threshold:
                is_1_normal = True
        # For large samples, assume normality (central limit theorem)
        else:
            is_1_normal = True

        # Check if dataset 2 is normally distributed
        if len(self.dataset_2[feat]) < 5000:
            # For small samples, use shapiro-wilk
            if len(self.dataset_2[feat]) < 50:
                _, p = stats.shapiro(self.dataset_2[feat])
            # For midsize samples, use kolmogorov-smirnov
            else:
                _, p = stats.kstest(self.dataset_2[feat], stats.norm.cdf)
            # If p < 0.05, reject hypothesis that data is normal
            if p >= self.pvalue_threshold:
                is_2_normal = True
        # For large samples, assume normality (central limit theorem)
        else:
            is_2_normal = True

        if self.verbose:
            if is_1_normal:
                print("Dataset 1 is normally distributed.")
            else:
                print("Dataset 1 is not normally distributed.")
            if is_2_normal:
                print("Dataset 2 is normally distributed.")
            else:
                print("Dataset 2 is not normally distributed.")

        return is_1_normal and is_2_normal

    def plot_hist(self, feat):
        # plot histogram of feature
        plt.figure(figsize=(10, 8))
        plt.ticklabel_format(style="scientific", axis="both")
        sns.histplot(
            data=self.dataset_1,
            x=feat,
            color="blue",
            label=self.dataset_names[0],
            kde=True,
            stat="density",
        )
        sns.histplot(
            data=self.dataset_2,
            x=feat,
            color="orange",
            label=self.dataset_names[1],
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.title(feat.upper())
        plt.show()

    def compare(self):
        if self.verbose:
            print("STARTING ANALYSIS")
        significant_combinations = []
        stats_all = []

        for feat in self.dataset_1.columns:
            if self.verbose:
                print(feat.upper())

            # If feature is a ventricle, transform to log
            if "ventricle" in feat:
                self.dataset_1[feat] = self.dataset_1[feat].apply(
                    lambda x: x if x == 0 else np.log(x)
                )
                self.dataset_2[feat] = self.dataset_2[feat].apply(
                    lambda x: x if x == 0 else np.log(x)
                )

            # Whether to check if data is normally distributed (only for small samples due to central limit theorem)
            if self.test_normalcy:
                is_normal = self.is_normal(feat)
            else:
                is_normal = True

            # Which test to use based on normalcy and independence
            if is_normal and self.ind:
                if self.verbose:
                    print("Data is normally distributed and independent.")
                    print("Using independent t-test to compare datasets.")
                _, p = stats.ttest_ind(self.dataset_1[feat], self.dataset_2[feat])
            elif is_normal and not self.ind:
                if self.verbose:
                    print("Data is normally distributed and dependent.")
                    print("Using dependent t-test to compare datasets.")
                _, p = stats.ttest_rel(self.dataset_1[feat], self.dataset_2[feat])

            # NOTE: These are non-parametric tests meaning they don't test mean
            elif not is_normal and self.ind:
                if self.verbose:
                    print("Data is independent but not normally distributed.")
                    print("Using Mann-Whitney U test to compare datasets.")
                _, p = stats.mannwhitneyu(self.dataset_1[feat], self.dataset_2[feat])
            elif not is_normal and not self.ind:
                if self.verbose:
                    print("Data is dependent but not normally distributed.")
                    print("Using Wilcoxon test to compare datasets.")
                _, p = stats.wilcoxon(self.dataset_1[feat], self.dataset_2[feat])

            stats_all.append([feat, p])
            if p < self.pvalue_threshold:
                significant_combinations.append([feat, p])

        all_stats = pd.DataFrame(stats_all, columns=["feature", "p_value"]).sort_values(
            by=["p_value"]
        )
        sig_vols = pd.DataFrame(
            significant_combinations, columns=["feature", "p_value"]
        ).sort_values(by=["p_value"])

        return sig_vols, all_stats

    def plot_feat(self, feat):
        # merge datasets to use seaborn
        self.dataset_1["dataset"] = self.dataset_names[0]
        self.dataset_2["dataset"] = self.dataset_names[1]
        df = pd.concat([self.dataset_1, self.dataset_2])

        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="dataset", y=feat)

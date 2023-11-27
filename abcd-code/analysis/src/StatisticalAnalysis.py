from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class StatisticalAnalysis:
    def __init__(
        self,
        dataset_1,
        dataset_2,
        scale=False,  # whether to standard scale the data
        dataset_names=["dataset_1", "dataset_2"],
        ind=True,  # if the datasets are independent
        verbose=False,
    ):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.ind = ind
        self.pvalue_threshold = 0.05
        self.verbose = verbose
        self.dataset_names = dataset_names
        if scale:
            self.scale_data()

    def scale_data(self):
        # standard scale the data
        scaler = StandardScaler()
        scaled_1 = scaler.fit_transform(self.dataset_1.to_numpy())
        scaled_2 = scaler.fit_transform(self.dataset_2.to_numpy())

        # convert back to dataframe
        self.dataset_1 = pd.DataFrame(scaled_1, columns=self.dataset_1.columns, index=self.dataset_1.index)
        self.dataset_2 = pd.DataFrame(scaled_2, columns=self.dataset_2.columns, index=self.dataset_2.index)


    def is_normal(self, feat):
        # Check if dataset 1 is normally distributed
        if len(self.dataset_1[feat]) < 5000:
            # For small samples, use shapiro-wilk
            if len(self.dataset_1[feat]) < 50:
                _, p = stats.shapiro(self.dataset_1[feat])
            # For midsize samples, use kolmogorov-smirnov
            else:
                _, p = stats.kstest(self.dataset_1[feat], stats.norm.cdf)
            # If p < 0.05, reject hypothesis that data is normal
            if p < self.pvalue_threshold:
                is_1_normal = False
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
            if p < self.pvalue_threshold:
                is_2_normal = False
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

        return (is_1_normal and is_2_normal)

    def plot_hist(self, feat):
        # plot histogram of feature
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.dataset_1, x=feat, color="blue", label=self.dataset_names[0], kde=True)
        sns.histplot(data=self.dataset_2, x=feat, color="orange", label=self.dataset_names[1], kde=True)
        plt.legend()
        plt.show()

    def compare_scaled(self):
        """Do statistical analysis on scaled dataset"""
        if self.verbose:
            print("STARTING ANALYSIS")
            print("Assuming data has been scaled to a normal distribution.")
        significant_combinations = []
        stats_all = []

        for feat in self.dataset_1.columns:
            if self.verbose:
                print(feat.upper())

            if self.ind:
                print("Using independent t-test to compare datasets.")
                _, p = stats.ttest_ind(self.dataset_1[feat], self.dataset_2[feat])
            else:
                print("Using dependent t-test to compare datasets.")
                _, p = stats.ttest_rel(self.dataset_1[feat], self.dataset_2)

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

    def compare(self):
        if self.verbose:
            print("STARTING ANALYSIS")
        significant_combinations = []
        stats_all = []

        for feat in self.dataset_1.columns:
            if self.verbose:
                print(feat.upper())

            # If not, use nonparametric test (wilcoxon for dependent samples)
            is_normal = self.is_normal(feat)
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

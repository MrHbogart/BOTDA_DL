import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Set global style for publication
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.format': 'svg',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
})


def compare_and_visualize(results_dict, save_dir=None, show=True):
    """
    Compare and visualize results from multiple approaches.
    Args:
        results_dict: dict of {approach_name: result_dict}
        save_dir: directory to save plots (optional)
        show: whether to display plots interactively
    """
    keys = set()
    for res in results_dict.values():
        keys.update(res.keys())
    keys = list(keys)
    palette = sns.color_palette('colorblind', len(results_dict))

    for key in keys:
        plt.figure(figsize=(8, 5))
        for idx, (name, res) in enumerate(results_dict.items()):
            if key in res:
                data = np.array(res[key])
                sns.histplot(data, kde=True, label=name, stat="density", element="step", fill=False, color=palette[idx], linewidth=2)
        plt.title(f"Comparison of {key}", fontsize=16)
        plt.xlabel(key, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(frameon=False, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/comparison_{key}.svg", format='svg')
            plt.savefig(f"{save_dir}/comparison_{key}.png", format='png', dpi=300)
        if show:
            plt.show()
        plt.close()

    # Boxplot for all keys
    for key in keys:
        plt.figure(figsize=(8, 5))
        data = [np.array(res[key]) for res in results_dict.values() if key in res]
        labels = [name for name, res in results_dict.items() if key in res]
        if len(data) > 1:
            bplot = plt.boxplot(data, labels=labels, patch_artist=True, medianprops=dict(color='black', linewidth=2))
            colors = sns.color_palette('colorblind', len(data))
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            plt.title(f"Boxplot of {key} across approaches", fontsize=16)
            plt.ylabel(key, fontsize=14)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/boxplot_{key}.svg", format='svg')
                plt.savefig(f"{save_dir}/boxplot_{key}.png", format='png', dpi=300)
            if show:
                plt.show()
            plt.close()


def statistical_tests(results_dict, key):
    """
    Perform pairwise statistical tests for a given key across all approaches.
    Returns a summary table of p-values.
    """
    names = list(results_dict.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = np.array(results_dict[names[i]][key]), np.array(results_dict[names[j]][key])
            # Use non-parametric test if data is not normal
            try:
                stat, p = ttest_ind(a, b, equal_var=False)
                test_name = 't-test'
            except Exception:
                stat, p = mannwhitneyu(a, b)
                test_name = 'Mann-Whitney U'
            results[(names[i], names[j])] = {'test': test_name, 'stat': stat, 'p-value': p}
    return results


def print_statistical_summary(results_dict, keys):
    for key in keys:
        print(f"\nStatistical tests for {key}:")
        stats = statistical_tests(results_dict, key)
        for pair, res in stats.items():
            print(f"{pair[0]} vs {pair[1]}: {res['test']} stat={res['stat']:.3f}, p-value={res['p-value']:.3e}")

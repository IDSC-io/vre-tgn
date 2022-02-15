# %%
# based on: https://github.com/hirsch-lab/roc-utils

import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
from sklearn import metrics
import scipy

show_details = True

sns.set_context("notebook")

from pathlib import Path

# %%
p = Path('../data/processed/')

tgn_fprs = []
tgn_tprs = []
tgn_aucs = []
for path in p.glob('**/*tgn_roc_auc.pickle'):
    with open(path, 'rb') as fh:
        tgn_roc_auc = pickle.load(fh)
        tgn_fpr, tgn_tpr, tgn_auc = tgn_roc_auc
        tgn_fprs.append(tgn_fpr)
        tgn_tprs.append(tgn_tpr)
        tgn_aucs.append(tgn_auc)


with open('../data/processed/20220211082623_tgn_roc_auc.pickle', 'rb') as fh:
    tgn_roc_auc = pickle.load(fh)
    tgn_fpr, tgn_tpr, tgn_auc_roc = tgn_roc_auc

mlp_fprs = []
mlp_tprs = []
mlp_aucs = []
for path in p.glob('*mlp_roc_auc.pickle'):
    with open(path, 'rb') as fh:
        mlp_roc_auc = pickle.load(fh)
        mlp_fpr, mlp_tpr, mlp_auc = mlp_roc_auc
        mlp_fprs.append(mlp_fpr)
        mlp_tprs.append(mlp_tpr)
        mlp_aucs.append(mlp_auc)

# %%

def get_mean_and_ci_of_rocs(tp_rates, fp_rates):
    n_samples = len(fp_rates)

    # resample for mean ROC and CIs
    longest_fpr = np.array([0, 1])
    for fpr in fp_rates:
        if len(longest_fpr) < len(fpr):
            longest_fpr = fpr

    longest_fpr = np.expand_dims(longest_fpr, axis=1)
    tp_rates_resampled = []
    fp_rates_resampled = []
    for tpr, fpr in zip(tp_rates, fp_rates):
        tpr_r = np.interp(longest_fpr, fpr, tpr)
        fp_rates_resampled.append(longest_fpr)
        tpr_r = np.expand_dims(tpr_r, axis=1)
        tp_rates_resampled.append(tpr_r)
    
    tprs = np.concatenate(tp_rates_resampled, axis=1)
    fprs = np.concatenate(fp_rates_resampled, axis=1)

    # get mean ROC
    tpr_mean = np.mean(tprs, axis=1)
    fpr_mean = np.mean(fprs, axis=1)

    # 95% confidence interval
    tpr_std = np.std(tprs, axis=1, ddof=1)
    tpr_lower_ci = tpr_mean - 1.96 * tpr_std / np.sqrt(n_samples)
    tpr_upper_ci = tpr_mean + 1.96 * tpr_std / np.sqrt(n_samples)

    return tpr_mean.squeeze(), fpr_mean.squeeze(), tpr_lower_ci.squeeze(), tpr_upper_ci.squeeze()


def get_optimal_operating_scheme(fprs1, tprs1, fprs2, tprs2):
    """
    based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3681096/
    """
    # determine higher and lower res ROCs
    high_res_fpr = fprs1 if len(fprs1) > len(fprs2) else fprs2
    high_res_tpr = tprs1 if len(tprs1) > len(tprs2) else tprs2
    low_res_fpr = fprs2 if len(fprs1) > len(fprs2) else fprs1
    low_res_tpr = tprs2 if len(tprs1) > len(tprs2) else tprs1

    # downsample the higher resolution ROC curve  
    equal_res_tpr = np.interp(low_res_fpr, high_res_fpr, high_res_tpr)
    
    combined_fprs = []
    combined_tprs = []
    for fpr1, tpr1, fpr2, tpr2 in zip(low_res_fpr, low_res_tpr, low_res_fpr, equal_res_tpr):
        
        conditions = [tpr1 * tpr2 - fpr1 * fpr2,
        tpr1 * (1 - tpr2) - fpr1 * ( 1 - fpr2),
        (1- tpr1) * tpr2 - (1 - fpr1) * fpr2,
        (1 - tpr1) * ( 1 - tpr2) - (1 - fpr1) * (1 - fpr2)]

        # calculate MLE estimate for estimator combinations
        fpr_estimates = [
            fpr1 * fpr2, # A and B
            fpr1, # A
            fpr2, # B
            fpr1 + fpr2 - fpr1 * fpr2 # A or B
        ]

        tpr_estimates = [
            tpr1 * tpr2, # A and B
            tpr1, # A
            tpr2, # B
            tpr1 + tpr2 - tpr1 * tpr2 # A or B
        ]

        #i = np.argmax(gains)
        combined_fprs.append(fpr_estimates)
        combined_tprs.append(tpr_estimates)

    combined = np.vstack([np.array(combined_fprs).flatten(), np.array(combined_tprs).flatten()])
    combined = np.concatenate([combined, np.array([[0], [1]])], axis=1).T
    last = combined.shape[0]

    hull = scipy.spatial.ConvexHull(points=combined, qhull_options=f'QG{last-1}')

    outer_fprs  = []
    outer_tprs = []
    for visible_facet in hull.simplices[hull.good]:
        outer_fprs.append(hull.points[visible_facet, 0])
        outer_tprs.append(hull.points[visible_facet, 1])

    outer_fprs, outer_tprs = np.array(outer_fprs), np.array(outer_tprs)
    sort_idc = np.argsort(outer_fprs, axis=0)
    bound_fprs, bound_tprs = np.take_along_axis(outer_fprs, sort_idc, axis=0)[:, 0], np.take_along_axis(outer_tprs, sort_idc, axis=0)[:, 0]
    bound_fprs = np.concatenate([np.array([0]), bound_fprs, np.array([1])])
    bound_tprs = np.concatenate([np.array([0]), bound_tprs, np.array([1])])
    return bound_fprs, bound_tprs



fig, ax = plt.subplots(1, figsize=(12, 8))

mlp_tpr_mean, mlp_fpr_mean, mlp_tpr_lower_ci, mlp_tpr_upper_ci = get_mean_and_ci_of_rocs(mlp_tprs, mlp_fprs)

ax.plot(mlp_fpr_mean, mlp_tpr_mean, lw=2, label=f"ROC curve of NN (AUC = {np.mean(mlp_aucs):0.2f})")

# print AUC confidence interval
auc_mean = np.mean(mlp_aucs)
auc_std = np.std(mlp_aucs, axis=0, ddof=1)
print(f"NN AUC {auc_mean} [{auc_mean - 1.96 * auc_std / np.sqrt(len(mlp_aucs))}-{auc_mean + 1.96 * auc_std / np.sqrt(len(mlp_aucs))}]")

tgn_tpr_mean, tgn_fpr_mean, tgn_tpr_lower_ci, tgn_tpr_upper_ci = get_mean_and_ci_of_rocs(tgn_tprs, tgn_fprs)

ax.plot(tgn_fpr_mean, tgn_tpr_mean, lw=2, label=f"ROC curve of TGN (AUC = {np.mean(tgn_aucs):0.2f})")

benefit_fprs, benefit_tprs = get_optimal_operating_scheme(mlp_fpr_mean, mlp_tpr_mean, tgn_fpr_mean, tgn_tpr_mean)
ax.plot(benefit_fprs, benefit_tprs, lw=2, linestyle="dashed", label=f"Estimated benefit of NN+TGN AUC = {metrics.auc(benefit_fprs, benefit_tprs):0.2f})")

ax.fill_between(mlp_fpr_mean, mlp_tpr_lower_ci, mlp_tpr_upper_ci,
                color="blue", alpha=.3,
                label="NN 95% CI")

# print AUC confidence interval
auc_mean = np.mean(tgn_aucs)
auc_std = np.std(tgn_aucs, axis=0, ddof=1)
print(f"TGN AUC {auc_mean} [{auc_mean - 1.96 * auc_std / np.sqrt(len(tgn_aucs))}-{auc_mean + 1.96 * auc_std / np.sqrt(len(tgn_aucs))}]")


ax.fill_between(tgn_fpr_mean, tgn_tpr_lower_ci, tgn_tpr_upper_ci,
                color="orange", alpha=.3,
                label="TGN 95% CI")

ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="no skill")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve of Patient VRE Risk Classification")
ax.legend(loc="lower right")
plt.savefig("../data/processed/roc_curve_nn_tgn.png")
plt.show()
# %%

# %%

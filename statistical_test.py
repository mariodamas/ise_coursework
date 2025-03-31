import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import shapiro
import pandas as pd

# Load the data
svm_we_f1_scores = []
svm_tf_f1_scores = []
nb_f1_scores = []

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
for project in projects:
    
    appendix = 'NB'
    nb = pd.read_csv(f"./results_{appendix}/{project}_{appendix}.csv", on_bad_lines='skip')
    nb_f1_scores.extend(nb['F1'].dropna().tolist())
    
    appendix = 'SVM_TF'
    svm_tf = pd.read_csv(f"./results_{appendix}/{project}_{appendix}.csv", on_bad_lines='skip')
    svm_tf_f1_scores.extend(svm_tf['F1'].dropna().tolist())
    
    appendix = 'SVM_WE'
    svm_we = pd.read_csv(f"./results_{appendix}/{project}_{appendix}.csv", on_bad_lines='skip')
    svm_we_f1_scores.extend(svm_we['F1'].dropna().tolist())

print(f"NB F1 scores: {nb_f1_scores}")
print(f"SVM_TF F1 scores: {svm_tf_f1_scores}")
print(f"SVM_WE F1 scores: {svm_we_f1_scores}")
print("\n")


# Shapiro-Wilk normality test
_, p_svm_we = shapiro(svm_we_f1_scores)
_, p_nb = shapiro(nb_f1_scores)
_, p_svm_tf = shapiro(svm_tf_f1_scores)

print("SVM_WORD EMBEDDED VS NAIVE BAYES")
if p_svm_we > 0.05 and p_nb > 0.05:
    print("The data are normal -> We use Paired t-test")
    t_stat, p_value = ttest_rel(svm_we_f1_scores, nb_f1_scores)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Statistically significant difference: SVM_WORD EMDEDDED is better than Naive Bayes")
    else:
        print("There is not enough evidence to claim that SVM_WORD EMDEDDED is better than Naive Bayes")

else:
    print("The data are NOT normal")
    
print("\n\n --- \n\n")

print("SVM_WORD EMDEDDED VS SVM_TERM FREQUENCY")
if p_svm_we > 0.05 and p_svm_tf > 0.05:
    print("The data are normal -> We use Paired t-test")
    t_stat, p_value = ttest_rel(svm_we_f1_scores, nb_f1_scores)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Statistically significant difference: SVM_WORD EMDEDDED is better than SVM_TERM FREQUENCY")
    else:
        print("There is not enough evidence to claim that SVM_WORD EMDEDDED is better than SVM_TERM FREQUENCY")

else:
    print("The data are NOT normal")

# Medians
median_svm_we = np.median(svm_we_f1_scores) if svm_we_f1_scores else np.nan
median_svm_tf = np.median(svm_tf_f1_scores) if svm_tf_f1_scores else np.nan
median_nb = np.median(nb_f1_scores) if nb_f1_scores else np.nan

medians = [
    ("SVM_WORD EMBEDDED", median_svm_we),
    ("SVM_TERM FREQUENCY", median_svm_tf),
    ("Naive Bayes", median_nb),
]

medians_sorted = sorted(medians, key=lambda x: x[1], reverse=True)
print("\n=== F1 Score Medians (Ordered) ===")
for label, median in medians_sorted:
    print(f"{label}: {median:.4f}")
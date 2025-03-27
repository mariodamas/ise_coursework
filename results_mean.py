import numpy as np
import pandas as pd

# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []
times       = []

# Choose the project (options: 'NB', 'SVM_TF', 'SVM_WE')
appendix = 'SVM_WE'
out_csv_name = f'./mean_results_{appendix}.csv'

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
required_columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Time']

for project in projects:
    try:
        print(f"results_{appendix}/{project}_{appendix}.csv")
        df = pd.read_csv(f"./results_{appendix}/{project}_{appendix}.csv", on_bad_lines='skip')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in {project}_{appendix}.csv: {missing_columns}")
            continue
        
        # Convert 'Time' column to seconds
        df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
        
        # Convert other columns to numeric, forcing errors to NaN
        df[required_columns[:-1]] = df[required_columns[:-1]].apply(pd.to_numeric, errors='coerce')
        
        # Append data to lists
        accuracies.extend(df['Accuracy'].dropna().tolist())
        precisions.extend(df['Precision'].dropna().tolist())
        recalls.extend(df['Recall'].dropna().tolist())
        f1_scores.extend(df['F1'].dropna().tolist())
        auc_values.extend(df['AUC'].dropna().tolist())
        times.extend(df['Time'].dropna().tolist())
    except pd.errors.ParserError as e:
        print(f"Error parsing {project}_{appendix}.csv: {e}")

# Ensuring lists are not empty before calculating mean
if accuracies:
    np_acc = np.mean(accuracies)
else:
    np_acc = np.nan

if precisions:
    np_prec = np.mean(precisions)
else:
    np_prec = np.nan

if recalls:
    np_rec = np.mean(recalls)
else:
    np_rec = np.nan

if f1_scores:
    np_f1 = np.mean(f1_scores)
else:
    np_f1 = np.nan

if auc_values:
    np_auc = np.mean(auc_values)
else:
    np_auc = np.nan

if times:
    np_time = np.mean(times)
else:
    np_time = np.nan

print(f"=== {appendix} Results ===")
print(f"Average Accuracy:      {np_acc:.4f}")
print(f"Average Precision:     {np_prec:.4f}")
print(f"Average Recall:        {np_rec:.4f}")
print(f"Average F1 score:      {np_f1:.4f}")
print(f"Average AUC:           {np_auc:.4f}")
print(f"Average Time:          {np_time:.4f} seconds")

# Save final results to CSV (overwrite mode)
df_log = pd.DataFrame(
    {
        'Accuracy': [np_acc],
        'Precision': [np_prec],
        'Recall': [np_rec],
        'F1': [np_f1],
        'AUC': [np_auc],
        'Time': [np_time],
    }
)

df_log.to_csv(out_csv_name, mode='w', header=True, index=False)

print(f"\nResults have been saved to: {out_csv_name}")
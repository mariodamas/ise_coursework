########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
time1 = pd.Timestamp.now()

# Classifier
from sklearn.svm import SVC

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)


########## 2. Load FastText Embeddings ##########

def load_fasttext_embeddings(filepath, embedding_dim=300): # 300 is the dimension of the FastText embeddings
    # the empty dictionary to store word embeddings
    embeddings_index = {}
    
    # Reading FastText embeddings file with UTF-8
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # skip the first line (usually contains metadata or header)
        
        for line in f:
            values = line.strip().split()
            word = values[0]  # The first element is the word
            vector = np.asarray(values[1:], dtype='float32')  # Converting the rest to a NumPy array of floats
            
            embeddings_index[word] = vector
    
    return embeddings_index

fasttext_path = 'cc.en.300.vec' # FastText embeddings downloaded from https://fasttext.cc/docs/en/crawl-vectors.html
embeddings_index = load_fasttext_embeddings(fasttext_path, embedding_dim=300)


########## 3. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Download & read data ##########
import os
import subprocess
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'pytorch'
path = f'datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Output CSV file name
out_csv_name = f'../{project}_SVM_WE.csv'

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# === Hyperparameter grid for SVM ===
param_grid = {                  
    'C': [0.01, 0.1, 1, 10, 100],     # Regularization parameter (maxime margin and minimizing classification errors)
    'kernel': ['linear',              # separate with a staight line
               'rbf']                 # separate with a curve if the data is not linearly separable
}


########## 5. Convert Text to FastText Embeddings ##########

def document_to_fasttext_vector(doc, embeddings_index, embedding_dim=300):
    words = doc.split()
    valid_embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if valid_embeddings:
        return np.mean(valid_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)

X_embeddings = np.array([document_to_fasttext_vector(doc, embeddings_index, 300) for doc in data[text_col]])

y = data['sentiment'].values
train_index, test_index = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42) # random_state needs to be the same number in order to get the same split
X_train, X_test = X_embeddings[train_index], X_embeddings[test_index]
y_train, y_test = y[train_index], y[test_index]
    
######### 6. SVM model & GridSearch ##########
clf = SVC(probability=True)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
  
# Retrieve the best model
best_clf = grid.best_estimator_
best_clf.fit(X_train, y_train)

# --- 4.4 Make predictions & evaluate ---
y_pred = best_clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Precision (macro)
prec = precision_score(y_test, y_pred, average='macro', zero_division=1)

# Recall (macro)
rec = recall_score(y_test, y_pred, average='macro')

# F1 Score (macro)
f1 = f1_score(y_test, y_pred, average='macro')


# AUC
# If labels are 0/1 only, this works directly.
# If labels are something else, adjust pos_label accordingly.
fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
auc_val = auc(fpr, tpr)

print(f"Project: {project}")
time_rest = pd.Timestamp.now() - time1
time_rest_seconds = time_rest.total_seconds()

print("=== SVM Model + Word Embeddings Results ===")
print(f"Average Accuracy:      {acc:.4f}")
print(f"Average Precision:     {prec:.4f}")
print(f"Average Recall:        {rec:.4f}")
print(f"Average F1 score:      {f1:.4f}")
print(f"Average AUC:           {auc_val:.4f}")
print(f"Time taken:            {time_rest_seconds:.4f} seconds")

# Save final results to CSV
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'Accuracy': [acc],
        'Precision': [prec],
        'Recall': [rec],
        'F1': [f1],
        'AUC': [auc_val],
        'Time': [time_rest],
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")
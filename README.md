# Bug Report Classification using SVM and FastText

## Project Overview
This project aims to classify bug reports as performance or non-performance bugs using Support Vector Machines (SVM) and FastText embeddings. The dataset consists of bug reports from multiple machine learning frameworks, including TensorFlow, PyTorch, Keras, Incubator-Mxnet and Caffe.

## Solution Approach
We considered multiple models, including SVM and Random Forest, but selected SVM due to its better handling of structured data. Additionally, we incorporated FastText word embeddings to improve text representation, allowing for better classification performance.

### Key Steps in Implementation
1. **Data Preprocessing:**
   - HTML tag removal
   - Emoji & special character handling
   - Lowercasing & punctuation normalization
   - Stopword removal

2. **Feature Extraction:**
   - FastText word embeddings (300-dimensional vectors per word)
   - Mean vector representation per bug report

3. **Training and Testing:**
   - Splitting data into 80% training and 20% testing
   - Using the "sentiment" column as the label

4. **SVM Model Application:**
   - Hyperparameter tuning with GridSearch (C and kernel selection)
   - F1-score optimization

## File Structure
```
├── baseline.py                    # Baseline model implementation
├── results_mean.py                 # Mean results calculation
├── statistical_test.py             # Statistical analysis
├── svm_tf_idf.py                   # SVM with TF-IDF features
├── svm_word_embeddings.py          # SVM with word embeddings
├── results_NB                      # Performance results from baseline
├── results_SVM_TF                  # Performance results from SVM + TF-IDF
├── results_SVM_WE                  # Performance results from SVM + FastText
├── datasets                        # Datasets folder
├── requirements.pdf                # Project dependencies
├── manual.pdf                      # Project documentation
├── replication.pdf                 # Instruction to replicate results
└── README.md                       # Project Introduction
```


## Instructions 
Please check manual.pdf or replication.pdf to follow setup and execution instructions.
 
## Authors
- Mario Damas

## References
- [1] FastText: https://fasttext.cc/docs/en/crawl-vectors.html
- [2] Scikit-Learn: https://scikit-learn.org/



# Bug Report Classification using SVM and FastText

## Project Overview
This project aims to classify bug reports as performance or non-performance bugs using Support Vector Machines (SVM) and FastText embeddings. The dataset consists of bug reports from multiple machine learning frameworks, including TensorFlow, PyTorch, Keras, and Caffe.

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
├── *.csv                           # Performance results from various ML frameworks
└── README.md                       # Project documentation
```

## Setup and Execution
1. Install dependencies (fasttext optional as I had to download pre-trained manually):
   ```bash
   pip install numpy pandas scikit-learn fasttext
   ```
2. Run the baseline model (for every project):
   ```bash
   python baseline.py
   ```
3. Run the SVM model with TF-IDF (for every project):
   ```bash
   python svm_tf_idf.py
   ```
3. Execute SVM classification with word embeddings (for every project):
   ```bash
   python svm_word_embeddings.py
   ```
4. Run results_mean to obtain the results table that appears in the document
   ```bash
   python results_mean.py
   ```
5. Finally, check whether the baseline is beaten with the statistical test.
   ```bash
   python results_mean.py
   ```

## Authors
- Mario Damas

## References
- [1] FastText: https://fasttext.cc/docs/en/crawl-vectors.html
- [2] Scikit-Learn: https://scikit-learn.org/

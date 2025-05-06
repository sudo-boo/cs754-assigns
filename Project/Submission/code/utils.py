import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hamming_loss
from scipy.sparse import lil_matrix
from tqdm import tqdm
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit
from scipy.linalg import hadamard
import os

# -----------------------------
# Parameters for eurlex
# -----------------------------
d = 3993
e = 234 # 3k*logd
k = 5
m = 240
n = 2500
nt = 500
p = 5000

# # -----------------------------
# # Parameters for rcv1x
# # -----------------------------
# d = 2456
# e = 234 # 3k*logd
# k = 10
# m = 240
# n = 2000
# nt = 501
# p = 6000


# # -----------------------------
# # Parameters for delicious
# # -----------------------------
# d = 983
# e = 248 # 3k*logd
# k = 12
# m = 240
# n = 1580
# nt = 393
# p = 500
GT_type = "sparse_rand"
seed = 42
np.random.seed(seed)



# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(path, n_samples, p, d):
    X = lil_matrix((n_samples, p), dtype=np.float32)
    Y = np.zeros((n_samples, d), dtype=np.int32)

    with open(path, 'r') as f:
        # header = f.readline().strip()
        # print("Header:", header)
        for i, line in tqdm(enumerate(f), total=n_samples):
            if i >= n_samples:
                break
            parts = line.strip().split()
            labels = parts[0].split(',')
            for l in labels:
                try:
                    idx = int(l)
                    if 0 <= idx < d:
                        Y[i, idx] = 1
                except ValueError:
                    pass  # skip malformed or non-integer label

            for item in parts[1:]:
                if ':' in item:
                    idx_val = item.split(':')
                    if len(idx_val) != 2:
                        continue
                    try:
                        idx, val = int(idx_val[0]), float(idx_val[1])
                        if 0 <= idx < p:
                            X[i, idx] = val
                    except ValueError:
                        continue

            if i == 0:
                print("Parsed labels:", labels)
                print("First 10 non-zero features:", [(idx, X[i, idx]) for idx in X[i].nonzero()[1][:10]])

    return X.tocsr(), Y



# -----------------------------
# GT Matrix Builder
# -----------------------------
def build_gt_matrix(d, m, method="sparse_rand"):
    A = np.zeros((m, d), dtype=int)
    s = int(np.ceil(np.log2(d)))  # ensure better disjunct properties
    for j in range(d):
        ones = np.random.choice(m, size=s, replace=False)
        A[ones, j] = 1
    return A


# -----------------------------
# MLGT Training (Algorithm 1)
# -----------------------------
from sklearn.linear_model import LogisticRegression

def train_classifiers(X, Y, A):
    Z = np.zeros((X.shape[0], A.shape[0]), dtype=int)
    for i in range(X.shape[0]):
        Z[i] = np.any(A[:, Y[i] == 1], axis=1).astype(int)

    classifiers = []
    for j in tqdm(range(A.shape[0])):
        clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        clf.fit(X.toarray(), Z[:, j])
        classifiers.append(clf)

    # print training error
    Y_pred = np.zeros((X.shape[0], A.shape[0]), dtype=int)
    for j in range(A.shape[0]):
        Y_pred[:, j] = classifiers[j].predict(X)
    Y_pred = (Y_pred @ A > 0).astype(int)
    print("Training Hamming Loss:", hamming_loss(Y, Y_pred))
    # print training precision
    for k in range(1, 11):
        print(f"Training Precision@{k}:", precision_at_k(Y, Y_pred, k))
    # print training precision at k avg
    print("Training Precision@k avg:", np.mean([precision_at_k(Y, Y_pred, k) for k in range(1, 11)]))
    return classifiers


# -----------------------------
# MLGT Prediction with Probabilities
# -----------------------------
def predict_all_scores(X, classifiers, A):
    m = len(classifiers)
    print("converting to dense in test")
    X_dense = X.toarray()  # Convert once before batch prediction
    X = X_dense
    print("converted to dense in test")

    Z_hat = np.zeros((X.shape[0], m))
    for j, clf in enumerate(classifiers):
        Z_hat[:, j] = clf.predict_proba(X)[:, 1]  # probability of label 1

    Y_scores = np.zeros((X.shape[0], A.shape[1]))
    for l in range(A.shape[1]):
        rows = np.where(A[:, l] == 1)[0]
        Y_scores[:, l] = np.sum(Z_hat[:, rows], axis=1)
    return Y_scores

# -----------------------------
# Thresholding for binary prediction
# -----------------------------
def threshold_predictions(Y_scores, A, e):
    Y_pred = np.zeros_like(Y_scores, dtype=int)
    for l in range(A.shape[1]):
        rows = np.where(A[:, l] == 1)[0]
        for i in range(Y_scores.shape[0]):
            support_misses = np.sum(Y_scores[i, rows] < 0.5)  # count probable 0s
            if support_misses < e / 2:
                Y_pred[i, l] = 1
    return Y_pred

# -----------------------------
# Evaluation
# -----------------------------
def precision_at_k(y_true, y_scores, k):
    precisions = []
    for yt, yp in zip(y_true, y_scores):
        topk = np.argsort(-yp)[:k]
        correct = yt[topk].sum()
        precisions.append(correct / k)
    return np.mean(precisions)


# -----------------------------
# CS Matrix Builder
# -----------------------------
def build_cs_matrix(d, m,mode):
    # Random Gaussian matrix scaled to unit variance
    if mode == "Gaussian":
        A = np.random.randn(m, d)
        A /= np.linalg.norm(A, axis=0)
    elif mode == "Hadamard":
        
         # Find smallest power of 2 ≥ d
        d_h = 2 ** int(np.ceil(np.log2(d)))
        H = hadamard(d_h)

        # Subsample m rows uniformly at random
        selected_rows = np.random.choice(d_h, size=m, replace=False)
        A = H[selected_rows, :d]  # Take first d columns if d < d_h

        # Normalize rows to unit norm
        A = A / np.sqrt(d)
    return A

# -----------------------------
# MLCS Training
# -----------------------------
def train_regressors(X, Y, A_cs, alpha=1.0):
    Z = Y @ A_cs.T  # Compressed labels
    regressors = []

    for j in tqdm(range(A_cs.shape[0])):
        reg = Ridge(alpha=alpha)
        reg.fit(X.toarray(), Z[:, j])
        regressors.append(reg)
    
    return regressors

# -----------------------------
# MLCS Prediction + Sparse Recovery
# -----------------------------
def predict_all_scores_cs(X, regressors, A_cs, k):
    X = X.toarray()
    m = len(regressors)

    Z_hat = np.zeros((X.shape[0], m))
    for j, reg in enumerate(regressors):
        Z_hat[:, j] = reg.predict(X)

    Y_scores = np.zeros((X.shape[0], A_cs.shape[1]))
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    for i in range(X.shape[0]):
        omp.fit(A_cs, Z_hat[i])
        y_hat = omp.coef_
        Y_scores[i] = y_hat

    return Y_scores

# -----------------------------
# Thresholding for MLCS
# -----------------------------
def threshold_predictions_cs(Y_scores, threshold=0.5):
    return (Y_scores >= threshold).astype(int)



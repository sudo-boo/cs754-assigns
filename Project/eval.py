
from utils import *
import matplotlib.pyplot as plt
import os



# -----------------------------
# Main
# -----------------------------
# dir = 'Delicious'
# dir = 'RCV1-x'
dir = 'Eurlex'
# file_train = os.path.join(dir+'/', 'train.txt')
# file_train = os.path.join(dir+'/', 'rcv1x_train.txt')
file_train = os.path.join(dir+'/', 'eurlex_train.txt')
# file_test  = os.path.join(dir+'/', 'test.txt')
# file_test  = os.path.join(dir+'/', 'rcv1x_test.txt')
file_test  = os.path.join(dir+'/', 'eurlex_test.txt')


print("Loading training data...")
X_train, Y_train = load_dataset(file_train, n, p, d)
print("Loading testing data...")
X_test, Y_test = load_dataset(file_test, nt, p, d)

# -----------------------------
# Final Evaluation and Plotting
# -----------------------------
import matplotlib.pyplot as plt
os.makedirs(f"results/{dir}/train", exist_ok=True)
os.makedirs(f"results/{dir}/test", exist_ok=True)

# Evaluate for k = 1 to 10
k_values = list(range(1, 11))

# ---- Train ----
print("Evaluating on training data...")
A_gt = build_gt_matrix(d, m)
classifiers_gt = train_classifiers(X_train, Y_train, A_gt)
Y_scores_gt_train = predict_all_scores(X_train, classifiers_gt, A_gt)

A_hadamard = build_cs_matrix(d, m, "Hadamard")
classifiers_h = train_regressors(X_train, Y_train, A_hadamard)
Y_scores_h_train = predict_all_scores_cs(X_train, classifiers_h, A_hadamard, k)

A_gauss = build_cs_matrix(d, m, "Gaussian")
classifiers_g = train_regressors(X_train, Y_train, A_gauss)
Y_scores_g_train = predict_all_scores_cs(X_train, classifiers_g, A_gauss, k)

# ---- Test ----
print("Evaluating on test data...")
Y_scores_gt_test = predict_all_scores(X_test, classifiers_gt, A_gt)
Y_scores_h_test = predict_all_scores_cs(X_test, classifiers_h, A_hadamard, k)
Y_scores_g_test = predict_all_scores_cs(X_test, classifiers_g, A_gauss, k)

# ---- Precision@k ----
def plot_precision_curves(Y_true, Y_gt, Y_h, Y_g, split):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [precision_at_k(Y_true, Y_gt, k) for k in k_values], marker='o', label="MLGT")
    plt.plot(k_values, [precision_at_k(Y_true, Y_h, k) for k in k_values], marker='o', label="MLCS-Hadamard")
    plt.plot(k_values, [precision_at_k(Y_true, Y_g, k) for k in k_values], marker='o', label="MLCS-Gaussian")
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title(f"{split.capitalize()} Precision@k")
    plt.grid()
    plt.legend()
    plt.savefig(f"results/{dir}/{split}/precision_at_k.png")
    plt.show()

# ---- Hamming Loss ----
def plot_hamming_loss_curves(Y_true, Y_gt, Y_h, Y_g, split):
    plt.figure(figsize=(10, 6))
    loss_gt = hamming_loss(Y_true, threshold_predictions(Y_gt, A_gt, e))
    loss_h  = hamming_loss(Y_true, threshold_predictions_cs(Y_h))
    loss_g  = hamming_loss(Y_true, threshold_predictions_cs(Y_g))
    plt.plot(k_values, [loss_gt] * len(k_values), marker='o', label="MLGT")
    plt.plot(k_values, [loss_h] * len(k_values), marker='o', label="MLCS-Hadamard")
    plt.plot(k_values, [loss_g] * len(k_values), marker='o', label="MLCS-Gaussian")
    plt.xlabel("k")
    plt.ylabel("Hamming Loss")
    plt.title(f"{split.capitalize()} Hamming Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"results/{dir}/{split}/hamming_loss.png")
    plt.show()

# Generate all plots
plot_precision_curves(Y_train, Y_scores_gt_train, Y_scores_h_train, Y_scores_g_train, split="train")
plot_hamming_loss_curves(Y_train, Y_scores_gt_train, Y_scores_h_train, Y_scores_g_train, split="train")

plot_precision_curves(Y_test, Y_scores_gt_test, Y_scores_h_test, Y_scores_g_test, split="test")
plot_hamming_loss_curves(Y_test, Y_scores_gt_test, Y_scores_h_test, Y_scores_g_test, split="test")

    
    

    

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.stats import t
import scienceplots

# TO DO:
# Write a DB func
X = np.genfromtxt("fwhm_data.txt")
min_max_scaler = StandardScaler(with_std=True)
X = min_max_scaler.fit_transform(X)

params = []
for epsilon in range(1, 11, 1):
    for samples in range(1, 11, 1):
        # eps 0.7 is good
        db = DBSCAN(eps=epsilon / 10, min_samples=samples).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        try:
            print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
            params.append([epsilon / 10, samples, metrics.silhouette_score(X, labels)])
        except ValueError:
            params.append([epsilon / 10, samples, 0])

params = np.array(params)
with plt.style.context("science"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(r"Silhouette goodness")
    ax.set_ylabel("Min Samples")
    ax.set_xlabel("epsilon")

    x = params[:, 0].reshape(10, 10)
    y = params[:, 1].reshape(10, 10)
    z = params[:, 2].reshape(10, 10)

    print(f"Max goodness is {z.max()}")
    levels = np.linspace(z.min(), z.max(), 7)
    CS = ax.contourf(x, y, z, levels=levels, cmap="plasma")
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel("Silhouette Coefficient")
    plt.savefig("Optimal_params.pdf", dpi=150)
    plt.show()

good_indexes = np.where((params[:, 2] == z.max()))
good_params = params[good_indexes][0]
print(good_params)

db = DBSCAN(eps=good_params[0], min_samples=int(10)).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


X = min_max_scaler.inverse_transform(X)

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

with plt.style.context("science"):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [plt.cm.Set1(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_ylabel(r"FWHM, \AA")
    ax.set_xlabel(r"Wavelength, \AA")

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.legend()
    plt.savefig("Cluter.pdf", dpi=150)

class_member_mask = labels == 0
cluster = X[class_member_mask & core_samples_mask]
fwhm_mean = np.mean(cluster[:, 1])
fwhm_std = np.std(cluster[:, 1])
print(f"cluster mean: {fwhm_mean}")
print(f"cluster std: {fwhm_std}")

res = stats.linregress(cluster[:, 0], cluster[:, 1])
print(f"R-squared: {res.rvalue**2:.6f}")
plt.plot(cluster[:, 0], cluster[:, 1], "o", label="original data")
plt.plot(
    cluster[:, 0], res.intercept + res.slope * cluster[:, 0], "r", label="fitted line"
)
plt.legend()

tinv = lambda p, df: abs(t.ppf(p / 2, df))

ts = tinv(0.05, len(x) - 2)
print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
print(f"intercept (95%): {res.intercept:.6f}" f" +/- {ts*res.intercept_stderr:.6f}")

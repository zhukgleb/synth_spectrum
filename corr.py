import numpy as np
import matplotlib.pyplot as plt
from data import extract_data, get_path2
from shifts import calculate_shift_index

p = get_path2("Test_spectrum.syn")
ang_1, flux_1, _ = extract_data(p)
ang_2, flux_2, _ = extract_data(p)

start_ang = 5000
end_ang = 5010

index_start_1 = np.where(ang_1==start_ang)[0][0]
index_end_1 = np.where(ang_1==end_ang)[0][0]
ang_1 = ang_1[index_start_1:index_end_1]
flux_1 = flux_1[index_start_1:index_end_1]

index_start_2 = np.where(ang_2==start_ang)[0][0]
index_end_2 = np.where(ang_2==end_ang)[0][0]
ang_2 = ang_2[index_start_2:index_end_2]
flux_2 = flux_2[index_start_2:index_end_2]

# Calculate shift index
ang_resolution = ang_1[1] - ang_1[0]
shift_index = calculate_shift_index(ang_resolution, 10000)
print(shift_index)

original_x = flux_1 
original_y = flux_2

# Create a shifted values
shifted_versions = [
    (original_x[shift_index:], original_y),
    (original_x, original_y[:]),
]

# Function to calculate correlation
def correlation(x, y):
    shortest = min(x.shape[0], y.shape[0])
    return np.corrcoef(x[:shortest], y[:shortest])[0, 1]

# Function to plot time series and show the correlation
def plot_correlation(x, y, text):
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.title(f"Correlation {text}: {correlation(x, y)}")
    plt.legend(loc="best")
    plt.show()

# Show results without shifting
for x, y in shifted_versions:
    plot_correlation(x, y, "before shifting")

from scipy.signal import correlate
from scipy.signal import correlation_lags

# Function to calculate cross-correlation,
# extract the best matching shift and then shift
# one of the series appropriately.
def shift_for_maximum_correlation(x, y):
    correlation_result = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation_result)]
    print(f"Best lag: {lag}")
    if lag < 0:
        y = y[-lag:]
    else:
        x = x[lag:]
    return x, y

# Plot results after shifting
for x, y in shifted_versions:
    shifted_x, shifted_y = shift_for_maximum_correlation(x, y)
    plot_correlation(shifted_x, shifted_y, text="after shifting")

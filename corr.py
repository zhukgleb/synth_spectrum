import numpy as np
import matplotlib.pyplot as plt
from data import test_data
from shifts import calculate_shift_index

start_ang = 5000
end_ang = 5010

ang_1, flux_1 = test_data()
ang_2, flux_2 = test_data()

# Calculate shift index
ang_resolution = ang_1[1] - ang_1[0]
shift_index = calculate_shift_index(ang_resolution, 10000)
print(shift_index)

original_series_1 = flux_1 
original_series_2 = flux_2

# Create a shifted values
shifted_versions = [
    (original_series_1[shift_index:], original_series_2)]

# Function to calculate correlation
def correlation(series_1: np.ndarray, series_2: np.ndarray):
    shortest = min(series_1.shape[0], series_2.shape[0])
    return np.corrcoef(series_1[:shortest], series_2[:shortest])[0, 1]


def plot_correlation(series_1: np.ndarray, series_2: np.ndarray, text: str):
    plt.figure(figsize=(10, 8))
    plt.plot(series_1, label="First spectrum")
    plt.plot(series_2, label="Second spectrum")
    plt.title(f"Correlation {text}: {correlation(series_1, series_2)}")
    plt.legend(loc="best")
    plt.show()

# Show results without shifting
for series_1, series_2 in shifted_versions:
    plot_correlation(series_1, series_2, "before shifting")

from scipy.signal import correlate
from scipy.signal import correlation_lags

# Function to calculate cross-correlation,
# extract the best matching shift and then shift
# one of the series appropriately.
def shift_for_maximum_correlation(series_1: np.ndarray, series_2: np.ndarray):
    correlation_result = correlate(series_1, series_2, mode="full")
    lags = correlation_lags(series_1.size, series_2.size, mode="full")
    lag = lags[np.argmax(correlation_result)]
    print(f"Best lag: {lag}")
    if lag < 0:
        series_2 = series_2[-lag:]
    else:
        series_1 = series_1[lag:]
    return series_1, series_2

# Plot results after shifting
for series_1, series_2 in shifted_versions:
    shifted_series_1, shifted_series_2 = shift_for_maximum_correlation(series_1, series_2)
    plot_correlation(shifted_series_1, shifted_series_2, text="after shifting")

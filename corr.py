import numpy as np
import matplotlib.pyplot as plt
from data import test_data, test_data_ideal
from shifts import calculate_shift_index, doppler_shift, calculate_doppler_from_shift
from scipy.interpolate import interp1d


start_ang = 4800
end_ang = 5400


ang_1_original, flux_1_original = test_data(start=start_ang, end=end_ang)
ang_2_original, flux_2_original = test_data_ideal(start=start_ang, end=end_ang)

# Raw part
velocity = 1000
ds = doppler_shift(velocity)
ang_resolution = ang_1_original[1] - ang_1_original[0]
print(f"doppler shift as {ds}, lambda res is {ang_resolution}")
interp_multi = round(ang_resolution / ds)
dots = int(len(ang_2_original)*interp_multi) * 10
ang_1 = np.linspace(np.min(ang_1_original), np.max(ang_1_original), dots)
ang_2 = np.linspace(np.min(ang_2_original), np.max(ang_2_original), dots)

f1 = interp1d(ang_1_original, flux_1_original, kind="quadratic")
f2 = interp1d(ang_2_original, flux_2_original, kind="quadratic")

flux_1 = f1(ang_1)
flux_2 = f2(ang_2)

# Calculate shift index
ang_resolution = ang_1[1] - ang_1[0]
shift_index = calculate_shift_index(ang_resolution, velocity)
print(shift_index)

original_series_1 = flux_1 
original_series_2 = flux_2

# Create a shifted values
shifted_versions = [
        (original_series_1, original_series_2[shift_index:])]

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
#for series_1, series_2 in shifted_versions:
#    plot_correlation(series_1, series_2, "before shifting")

from scipy.signal import correlate
from scipy.signal import correlation_lags

# Function to calculate cross-correlation,
# extract the best matching shift and then shift
# one of the series appropriately.
def shift_for_maximum_correlation(series_1: np.ndarray, series_2: np.ndarray):
    correlation_result = correlate(series_1, series_2, mode="full")
    lags = correlation_lags(series_1.size, series_2.size, mode="full")
    lag = lags[np.argmax(correlation_result)]

#    lag_corr_arr = np.column_stack((lags, correlation_result))
#    sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]
#    plt.title("Lag-correlation")
#    plt.plot(sortd_lag_corr_arr[:, 0], sortd_lag_corr_arr[:, 1])
#    plt.show()

    print(f"Best lag: {lag}")
    if lag < 0:
        series_2 = series_2[-lag:]
    else:
        series_1 = series_1[lag:]
    return series_1, series_2, lag


# Plot results after shifting
for series_1, series_2 in shifted_versions:
    shifted_series_1, shifted_series_2, lag = shift_for_maximum_correlation(series_1, series_2)
    # plot_correlation(shifted_series_1, shifted_series_2, text="after shifting")
    lag_ang = calculate_doppler_from_shift(ang_resolution * lag)
    print(f"lag in meters is : {lag_ang}")


# Astropy Correlation
from specutils.analysis import template_correlate
from specutils import Spectrum1D
from astropy import units as u

ang_1, ang_2 = ang_1 * u.angstrom, ang_2 * u.angstrom

flux_1, flux_2 = flux_1 * u.dimensionless_unscaled, flux_2 * u.dimensionless_unscaled
spectrum_obs = Spectrum1D(spectral_axis=ang_1, flux=flux_1)
spectrum_template = Spectrum1D(spectral_axis=ang_2, flux=flux_2)
correlate, lags = template_correlate(spectrum_obs, spectrum_template)

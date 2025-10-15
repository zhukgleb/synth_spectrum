import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline, interp1d


def spectrum_finder(folder_data: list) -> list:
    spectrum_list = []
    for i in range(len(folder_data)):
        if str(folder_data[i]).find(".spec") != -1:
            spectrum_list.append(str(folder_data[i]))
    return spectrum_list

def data_graber(path2spectrum: Path) -> np.ndarray:
    data = np.genfromtxt(path2spectrum, comments="#")
    return data

def read_spectrum_grid(path2grid: Path) -> list:
    folder_data = os.listdir(path2grid)
    spectrum_list = spectrum_finder(folder_data)

    data = []
    for i in range(len(spectrum_list)):
        data.append(data_graber(path2grid / spectrum_list[i]))
    return data

def deriv_flux(normal_flux: np.ndarray) -> np.ndarray:
    delta_arr = []
    delta = 2
    for i in range(1, len(normal_flux)):
        delta_arr.append((abs(normal_flux[i] - normal_flux[i - 1])) / delta)
        
    delta_arr.append(1)
    return delta_arr


class AutomaticContinuumFitter:
    def __init__(self, model_grid_wavelength, model_grid_spectra, 
                 model_params, continuum_degree:int=3, n_iterations:int=10):
        """
        model_grid_wavelength: array, common for models
        model_grid_spectra: 2D array, model spectra [n_models, n_wavelengths]
        model_params: 2D array, model params [Teff, logg, Fe/H, ...]
        continuum_degree: int, continuum poly degree
        """
        self.wavelength = model_grid_wavelength
        self.model_spectra = model_grid_spectra
        self.model_params = model_params
        self.continuum_degree = continuum_degree
        self.n_iterations = n_iterations
        
        # get normalize
        self.normalized_models = self._normalize_spectra(self.model_spectra)
        
        # find nearest parameter solution (working not good for now...)
        self.nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nbrs.fit(model_params)
    
    def _normalize_spectra(self, spectra):
        normalized = np.zeros_like(spectra)
        for i, spec in enumerate(spectra):
            window_size = len(spec) // 20  # just emperical...
            continuum = self._moving_max(spec, window_size)
            normalized[i] = spec / continuum
        return normalized
    
    def _moving_max(self, x, window_size):
        half_window = window_size // 2
        continuum = np.zeros_like(x)
        
        for i in range(len(x)):
            start = max(0, i - half_window)
            end = min(len(x), i + half_window)
            continuum[i] = np.max(x[start:end])
        
        # Spline smoothing
        indices = np.linspace(0, len(x)-1, min(50, len(x)//10), dtype=int)
        spline = CubicSpline(self.wavelength[indices], continuum[indices])
        return spline(self.wavelength)
    
    def _find_best_matching_model(self, observed_spec, current_continuum):
        if current_continuum is None:
            current_continuum = np.ones_like(observed_spec)
        
        normalized_observed = observed_spec / current_continuum
        
        distances = np.sqrt(np.sum((self.normalized_models - normalized_observed)**2, axis=1))
        best_idx = np.argmin(distances)
        
        return best_idx, distances[best_idx]
    
    def _fit_continuum_to_model(self, observed_spec, model_spec):
        # observed_spec / continuum ≈ model_spec
        # => continuum \approx observed_spec / model_spec
        
        raw_continuum_estimate = observed_spec / model_spec
        
        # ignore lines
        median_cont = np.median(raw_continuum_estimate)
        mad_cont = np.median(np.abs(raw_continuum_estimate - median_cont))
        
        # make a mask without lines
        continuum_mask = (raw_continuum_estimate > median_cont - 2*mad_cont)
        
        if np.sum(continuum_mask) < self.continuum_degree + 1:
            continuum_mask = np.ones_like(continuum_mask, dtype=bool)
        
        # poly fit
        x_normalized = np.linspace(0, 1, len(self.wavelength))
        coeffs = np.polyfit(x_normalized[continuum_mask], 
                           raw_continuum_estimate[continuum_mask], 
                           self.continuum_degree)
        
        continuum_poly = np.polyval(coeffs, x_normalized)
        
        # is it positive?
        continuum_poly = np.maximum(continuum_poly, 0.1 * np.median(observed_spec))
        
        return continuum_poly
    
    def fit(self, observed_wavelength, observed_spectrum, initial_continuum=None):

        interp_func = interp1d(observed_wavelength, observed_spectrum, 
                              kind='linear', bounds_error=False, 
                              fill_value='extrapolate')
        observed_interp = interp_func(self.wavelength)
        
        if initial_continuum is None:
            current_continuum = np.ones_like(observed_interp)
        else:
            interp_cont = interp1d(observed_wavelength, initial_continuum,
                                 kind='linear', bounds_error=False,
                                 fill_value='extrapolate')
            current_continuum = interp_cont(self.wavelength)
        
        history = {
            'continuums': [],
            'best_model_indices': [],
            'distances': [],
            'normalized_spectra': [],
            'best_params': []
        }
        
        print("Starting iterative continuum fitting...")
        
        for iteration in range(self.n_iterations):
            # Step 1: best matching model spectra
            best_idx, distance = self._find_best_matching_model(
                observed_interp, current_continuum)
            
            # Step 2: Best continuum from model spectra
            new_continuum = self._fit_continuum_to_model(
                observed_interp, self.normalized_models[best_idx])
            
            # Step 3: Smoothing new continuum
            if iteration > 0:
                alpha = 0.7  # Emperical...
                current_continuum = alpha * current_continuum + (1-alpha) * new_continuum
            else:
                current_continuum = new_continuum
            
            
            history['continuums'].append(current_continuum.copy())
            history['best_model_indices'].append(best_idx)
            history['distances'].append(distance)
            history['normalized_spectra'].append(observed_interp / current_continuum)
            history['best_params'].append(self.model_params[best_idx].copy())
            
            print(f"Iteration {iteration+1}: Best model index = {best_idx}, "
                  f"Distance = {distance:.6f}, "
                  f"Params = {self.model_params[best_idx]}")
            
            # Stop creteria
            if iteration > 1 and abs(history['distances'][-2] - distance) < 1e-6:
                print(f"Converged after {iteration+1} iterations")
                break
        
        # Interpolate again to the old grid
        final_continuum_interp = interp1d(self.wavelength, current_continuum,
                                        kind='cubic', bounds_error=False,
                                        fill_value='extrapolate')
        final_continuum = final_continuum_interp(observed_wavelength)
        
        return final_continuum, history


if __name__ == "__main__":
    pass
    # deriv_map(Path("data/2025-10-14-20-15-43_0.6420588332457386_NLTE_synthetic_spectra_parameters/"))

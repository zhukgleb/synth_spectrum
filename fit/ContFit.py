import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline, interp1d


class AutomaticContinuumFitter:
    def __init__(self, model_grid_wavelength, model_grid_spectra, 
                 model_params, continuum_degree=3, n_iterations=10):
        """
        model_grid_wavelength: array, common grid 
        model_grid_spectra: 2D array,  [n_models, n_wavelengths]
        model_params: 2D array, [Teff, logg, Fe/H, ...]
        continuum_degree: poly fit degree
        """
        self.wavelength = model_grid_wavelength
        self.model_spectra = model_grid_spectra
        self.model_params = model_params
        self.continuum_degree = continuum_degree
        self.n_iterations = n_iterations
        
        self.normalized_models = self._normalize_spectra(self.model_spectra)
        
        self.nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nbrs.fit(model_params)
    
    def _normalize_spectra(self, spectra):
        normalized = np.zeros_like(spectra)
        for i, spec in enumerate(spectra):
            window_size = len(spec) // 20  
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

        raw_continuum_estimate = observed_spec / model_spec
        
        median_cont = np.median(raw_continuum_estimate)
        mad_cont = np.median(np.abs(raw_continuum_estimate - median_cont))
        
        continuum_mask = (raw_continuum_estimate > median_cont - 2*mad_cont)
        
        if np.sum(continuum_mask) < self.continuum_degree + 1:
            continuum_mask = np.ones_like(continuum_mask, dtype=bool)
        
        x_normalized = np.linspace(0, 1, len(self.wavelength))
        coeffs = np.polyfit(x_normalized[continuum_mask], 
                           raw_continuum_estimate[continuum_mask], 
                           self.continuum_degree)
        
        continuum_poly = np.polyval(coeffs, x_normalized)
        
        continuum_poly = np.maximum(continuum_poly, 0.1 * np.median(observed_spec))
        
        return continuum_poly
    
    def fit(self, observed_wavelength, observed_spectrum, initial_continuum=None):
        """
        Основной итерационный алгоритм подбора континуума
        """
        obs_min, obs_max = observed_wavelength.min(), observed_wavelength.max()
        model_min, model_max = self.wavelength.min(), self.wavelength.max()
        

        overlap_min = max(obs_min, model_min)
        overlap_max = min(obs_max, model_max)
        
        if overlap_min >= overlap_max:
            raise ValueError("Нет перекрытия между наблюдаемым и модельным спектрами!")
        
        print(f"Область перекрытия спектров: [{overlap_min:.1f}, {overlap_max:.1f}] Å")
        

        obs_mask = (observed_wavelength >= overlap_min) & (observed_wavelength <= overlap_max)
        model_mask = (self.wavelength >= overlap_min) & (self.wavelength <= overlap_max)
        

        obs_wavelength_overlap = observed_wavelength[obs_mask]
        obs_spectrum_overlap = observed_spectrum[obs_mask]
        model_wavelength_overlap = self.wavelength[model_mask]
        
        interp_func = interp1d(obs_wavelength_overlap, obs_spectrum_overlap, 
                              kind='linear', bounds_error=False, 
                              fill_value='extrapolate')
        observed_interp = interp_func(model_wavelength_overlap)
        
        wavelength_overlap = model_wavelength_overlap
        models_overlap = self.model_spectra[:, model_mask]
        normalized_models_overlap = self.normalized_models[:, model_mask]
        
        if initial_continuum is None:
            current_continuum = np.ones_like(observed_interp)
        else:
            interp_cont = interp1d(observed_wavelength, initial_continuum,
                                 kind='linear', bounds_error=False,
                                 fill_value='extrapolate')
            current_continuum = interp_cont(wavelength_overlap)
        
        history = {
            'continuums': [],
            'best_model_indices': [],
            'distances': [],
            'normalized_spectra': [],
            'best_params': [],
            'wavelength_overlap': wavelength_overlap  # Сохраняем для визуализации
        }
        
        print("Starting iterative continuum fitting...")
        
        for iteration in range(self.n_iterations):
            best_idx, distance = self._find_best_matching_model_overlap(
                observed_interp, current_continuum, normalized_models_overlap)
            
            new_continuum = self._fit_continuum_to_model_overlap(
                observed_interp, normalized_models_overlap[best_idx])
            
            if iteration > 0:
                alpha = 0.7  
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
            
            if iteration > 1 and abs(history['distances'][-2] - distance) < 1e-6:
                print(f"Converged after {iteration+1} iterations")
                break
        
        final_continuum = self._extend_continuum_to_full_range(
            wavelength_overlap, current_continuum, observed_wavelength, 
            observed_spectrum, models_overlap[best_idx])
        
        return final_continuum, history
    
    def _find_best_matching_model_overlap(self, observed_spec, current_continuum, normalized_models_overlap):
        if current_continuum is None:
            current_continuum = np.ones_like(observed_spec)
        
        normalized_observed = observed_spec / current_continuum
        
        distances = np.sqrt(np.sum((normalized_models_overlap - normalized_observed)**2, axis=1))
        best_idx = np.argmin(distances)
        
        return best_idx, distances[best_idx]
    
    def _fit_continuum_to_model_overlap(self, observed_spec, model_spec):
        raw_continuum_estimate = observed_spec / model_spec
        
        median_cont = np.median(raw_continuum_estimate)
        mad_cont = np.median(np.abs(raw_continuum_estimate - median_cont))
        
        continuum_mask = (raw_continuum_estimate > median_cont - 2*mad_cont)
        
        if np.sum(continuum_mask) < self.continuum_degree + 1:
            continuum_mask = np.ones_like(continuum_mask, dtype=bool)
        
        x_normalized = np.linspace(0, 1, len(raw_continuum_estimate))
        coeffs = np.polyfit(x_normalized[continuum_mask], 
                           raw_continuum_estimate[continuum_mask], 
                           self.continuum_degree)
        
        continuum_poly = np.polyval(coeffs, x_normalized)
        continuum_poly = np.maximum(continuum_poly, 0.1 * np.median(observed_spec))
        
        return continuum_poly
    
    def _extend_continuum_to_full_range(self, overlap_wavelength, overlap_continuum, 
                                      full_wavelength, full_spectrum, best_model_overlap):
        
        continuum_interp = interp1d(overlap_wavelength, overlap_continuum,
                                  kind='cubic', bounds_error=False, 
                                  fill_value='extrapolate')
        continuum_full = continuum_interp(full_wavelength)
        
        obs_min, obs_max = full_wavelength.min(), full_wavelength.max()
        overlap_min, overlap_max = overlap_wavelength.min(), overlap_wavelength.max()
        
        left_mask = full_wavelength < overlap_min
        if np.any(left_mask):
            left_value = overlap_continuum[0]
            continuum_full[left_mask] = left_value
        

        right_mask = full_wavelength > overlap_max
        if np.any(right_mask):
            right_value = overlap_continuum[-1]
            continuum_full[right_mask] = right_value
        
        continuum_full = np.maximum(continuum_full, 0.1 * np.median(full_spectrum))
        
        window = min(50, len(continuum_full) // 10)
        if window > 1:
            from scipy.ndimage import uniform_filter1d
            continuum_full = uniform_filter1d(continuum_full, size=window)
        
        return continuum_full

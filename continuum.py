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

def parse_header_params(filename):
    params = {}
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                clean_line = line.strip('#').strip()
                if ':' in clean_line:
                    key, value = clean_line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Обработка числовых параметров
                    if key in ['teff', 'logg', '[Fe/H]', 'vmic', 'vmac', 'resolution', 'rotation']:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value
                    elif key in ['nlte_flag']:
                        params[key] = value.lower() == 'true'
                    else:
                        params[key] = value
            else:
                break
    
    return params

def load_spectrum(filename):
    params = parse_header_params(filename)
    
    # Загружаем данные, пропуская строки с комментариями
    try:
        data = np.loadtxt(filename, comments='#')
        if data.ndim == 1:
            # Если только одна строка данных
            wavelength = np.array([data[0]])
            flux = np.array([data[1]])
        else:
            wavelength = data[:, 0]  
            flux = data[:, 1]        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        data = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue
        data = np.array(data)
        if len(data) > 0:
            wavelength = data[:, 0]
            flux = data[:, 1]
        else:
            raise ValueError(f"No data found in {filename}")
    
    return wavelength, flux, params

def create_model_grid():
    model_files = ['1.spec', '2.spec', '3.spec', '4.spec', '5.spec']
    model_spectra = []
    model_wavelengths = []
    model_params_list = []
    
    print("Loading model spectra...")
    for i, filename in enumerate(model_files):
        if os.path.exists(filename):
            wl, flux, params = load_spectrum(filename)
            model_spectra.append(flux)
            model_wavelengths.append(wl)
            
            teff = params.get('teff', 5500)
            logg = params.get('logg', 4.0)
            feh = params.get('[Fe/H]', 0.0)
            
            model_params_list.append([teff, logg, feh])
            
            print(f"Loaded {filename}:")
            print(f"  Points: {len(wl)}, Teff: {teff}, logg: {logg}, [Fe/H]: {feh}")
            print(f"  Flux range: [{flux.min():.3f}, {flux.max():.3f}]")
        else:
            print(f"Warning: File {filename} not found")
            continue
    
    if not model_spectra:
        raise ValueError("No model spectra found!")
    
    # is it simular wavelenght grid?
    first_wl = model_wavelengths[0]
    for i, wl in enumerate(model_wavelengths[1:], 1):
        if len(wl) != len(first_wl) or not np.allclose(wl, first_wl, rtol=1e-6):
            print(f"Warning: Wavelength scales differ for model {i+1}. Interpolating...")
            interp_func = interp1d(wl, model_spectra[i], kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            model_spectra[i] = interp_func(first_wl)
    
    common_wavelength = first_wl
    model_spectra = np.array(model_spectra)
    model_params = np.array(model_params_list)
    
    return common_wavelength, model_spectra, model_params

def load_observed_spectrum(filename):
    print(f"Loading observed spectrum {filename}...")
    
    try:
        data = np.loadtxt(filename, comments='#')
        if data.ndim == 1:
            wavelength = np.array([data[0]])
            flux = np.array([data[1]])
        else:
            wavelength = data[:, 0]
            flux = data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        data = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue
        data = np.array(data)
        if len(data) > 0:
            wavelength = data[:, 0]
            flux = data[:, 1]
        else:
            raise ValueError(f"No data found in {filename}")
    
    print(f"Observed spectrum: {len(wavelength)} points, flux range [{flux.min():.3f}, {flux.max():.3f}]")
    
    return wavelength, flux

def main(save=False):
    model_wavelength, model_spectra, model_params = create_model_grid()  # Re-write to the any custom grid..
    n_models = len(model_spectra)
    
    # data = np.genfromtxt("0.spec", comments="#")
    data = np.genfromtxt("test_distorted_simple.spec", comments="#")
    obs_wavelength = data[:, 0]
    obs_spectrum = data[:, 1]
    del data
    
    fitter = AutomaticContinuumFitter(
        model_wavelength, 
        model_spectra, 
        model_params,
        continuum_degree=4, 
        n_iterations=30
    )
    
    print("\nStarting continuum fitting...")
    final_continuum, history = fitter.fit(obs_wavelength, obs_spectrum)
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(obs_wavelength, obs_spectrum, 'k-', label='obs spectra', alpha=0.7, linewidth=1)
    plt.plot(obs_wavelength, final_continuum, 'r-', linewidth=2, label='finded continuum')
    plt.xlabel('wavelenght, angstroms')
    plt.ylabel('flux')
    plt.legend()
    plt.title('Obs spectra and continuum')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    normalized_observed = obs_spectrum / final_continuum
    best_model_idx = history['best_model_indices'][-1]
    best_model_spectrum = model_spectra[best_model_idx]
    best_model_params = model_params[best_model_idx]
    
    plt.plot(obs_wavelength, normalized_observed, 'r-', label='normalized obs', alpha=0.8, linewidth=1)
    plt.plot(model_wavelength, best_model_spectrum, 'b--', label=f'best model (Teff={best_model_params[0]:.0f})', alpha=0.8)
    plt.xlabel('wavelenght (Å)')
    plt.ylabel('normalized flux')
    plt.legend()
    plt.title('best fit comp')
    plt.grid(True, alpha=0.3)
    
    # 3. Сходимость алгоритма
    plt.subplot(2, 2, 3)
    plt.plot(history['distances'], 'bo-', markersize=4)
    plt.xlabel('iteration')
    plt.ylabel('distance to model')
    plt.title('algoritm spread')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    model_interp = interp1d(model_wavelength, best_model_spectrum, 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    best_model_interp = model_interp(obs_wavelength)
    
    difference = normalized_observed - best_model_interp
    plt.plot(obs_wavelength, difference, 'g-', alpha=0.7, linewidth=1)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('wavelenght, ang')
    plt.ylabel('difference (obs - model)')
    plt.title('Difference between spectra')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig('continuum_fitting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    if save:
        print("\nSaving results...")
    
    normalized_data = np.column_stack([obs_wavelength, normalized_observed])
    header = f"# Normalized spectrum\n# Original: 0.spec\n# Continuum fitted automatically\n# Best matching model: Teff={best_model_params[0]:.0f}, logg={best_model_params[1]:.2f}, [Fe/H]={best_model_params[2]:.2f}"
    if save:
        np.savetxt('0_normalized.spec', normalized_data, 
                   header=header, fmt='%.6f')
    
    continuum_data = np.column_stack([obs_wavelength, final_continuum])
    header_cont = f"# Fitted continuum\n# Original: 0.spec\n# Best matching model: Teff={best_model_params[0]:.0f}, logg={best_model_params[1]:.2f}, [Fe/H]={best_model_params[2]:.2f}"
    if save:
        np.savetxt('0_continuum.spec', continuum_data,
                header=header_cont, fmt='%.6f')
    
    best_final_idx = history['best_model_indices'][-1]
    best_final_params = history['best_params'][-1]
    
    print(f"\n=== RESULTS ===")
    print(f"Best matching model: {best_final_idx + 1}.spec")
    print(f"Best parameters: Teff={best_final_params[0]:.1f} K, "
          f"logg={best_final_params[1]:.2f}, [Fe/H]={best_final_params[2]:.2f}")
    print(f"Final distance: {history['distances'][-1]:.6f}")
    print(f"Number of iterations: {len(history['distances'])}")
    
    print(f"\nResults saved to:")
    print(f"- 0_normalized.spec (normalized observed spectrum)")
    print(f"- 0_continuum.spec (fitted continuum)")
    print(f"- continuum_fitting_results.png (plots)")
    
    return final_continuum, history

if __name__ == "__main__":
    main()
    # deriv_map(Path("data/2025-10-14-20-15-43_0.6420588332457386_NLTE_synthetic_spectra_parameters/"))

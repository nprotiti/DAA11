import numpy as np
import matplotlib.pyplot as plt
import os
from astroML.time_series import lomb_scargle, lomb_scargle_bootstrap, lomb_scargle_BIC
import feets
import pandas as pd

# Function to load TESS light curve data from .lc files
def load_light_curve(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    time = data[:, 0]
    flux = data[:, 1]
    return time, flux

# Function to apply Fourier transformation
def apply_fourier_transformation(time, flux):
    y_fft = np.fft.fft(flux)
    k_values = [1, 3, 8]
    y_fits = {}
    for k in k_values:
        y_fft_copy = np.copy(y_fft)
        y_fft_copy[k + 1:-k] = 0
        y_fits[k] = np.fft.ifft(y_fft_copy).real
    return y_fits

# Function to apply Lomb-Scargle Periodogram
def apply_lomb_scargle(time, flux):
    N = len(time)
    dy = 0.1 + 0.1 * np.random.random(N)
    y_obs = np.random.normal(flux, dy)
    period = 10 ** np.linspace(-1, 0, 10000)
    omega = 2 * np.pi / period
    PS = lomb_scargle(time, y_obs, dy, omega, generalized=True)
    D = lomb_scargle_bootstrap(time, y_obs, dy, omega, generalized=True, N_bootstraps=1000, random_state=0)
    sig1, sig5 = np.percentile(D, [99, 99.9])
    BIC = lomb_scargle_BIC(PS, y_obs, dy)
    return PS, sig1, sig5, BIC

# Function to extract features using feets
def extract_features(time, flux):
    extractor = feets.Extractor()
    fs = extractor.extract(time=time, magnitude=flux)
    return dict(zip(fs[0], fs[1]))

# Function to save results
def save_results(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

# Main analysis function
def analyze_tess_light_curves(data_folder, output_file):
    results = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.lc'):
            file_path = os.path.join(data_folder, file_name)
            try:
                time, flux = load_light_curve(file_path)

                # Fourier transformation
                y_fits = apply_fourier_transformation(time, flux)

                # Lomb-Scargle Periodogram
                PS, sig1, sig5, BIC = apply_lomb_scargle(time, flux)

                # Feature extraction
                features = extract_features(time, flux)

                # Collecting results
                result = {
                    'file_name': file_name,
                    'PS': PS.tolist(),
                    'sig1': sig1,
                    'sig5': sig5,
                    'BIC': BIC.tolist(),
                    **features
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Save all results to an output file
    save_results(results, output_file)

if __name__ == "__main__":
    # Path to the TESS light curve data folder
    data_folder = '/home/user/Downloads/DAA11/_TESS_lightcurves_raw/ACV'
    output_file = 'tess_analysis_results.csv'

    # Run the analysis on all light curves and save results to a CSV file
    analyze_tess_light_curves(data_folder, output_file)

    # Load and display the results
    try:
        results = pd.read_csv(output_file)
        if not results.empty:
            print(results.head())
        else:
            print("No data in the output file.")
    except pd.errors.EmptyDataError:
        print("No data in the output file.")


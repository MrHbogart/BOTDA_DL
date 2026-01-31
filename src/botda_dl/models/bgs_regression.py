"""
BGS (Brillouin Gain Spectrum) Regression analysis module.
Implements regression-based peak frequency and FWHM estimation.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from tqdm import tqdm
import pickle
import random

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.regularizers import l2

from ..core import BaseAnalyzer


class BGSRegression(BaseAnalyzer):
    """
    BGS Regression-based analyzer for peak frequency and FWHM estimation.
    Uses Lorentzian fitting for spectral analysis.
    """

    def lorentzian(self, x, A, peak_x, gamma, offset):
        """Lorentzian function model."""
        return offset + (A * gamma**2) / ((x - peak_x)**2 + gamma**2)

    def single_lorentzian_fit(self, x, y):
        """Fit a single Lorentzian curve to data."""
        try:
            A_guess = np.max(y)
            peak_x_guess = x[x.shape[0]//2]
            gamma_guess = x[-x.shape[0]//4] - x[x.shape[0]//4]
            offset_guess = np.min(y)
            
            params, _ = curve_fit(
                self.lorentzian, x, y,
                p0=[A_guess, peak_x_guess, gamma_guess, offset_guess],
                maxfev=5000
            )
        except RuntimeError:
            print('Lorentzian fit failed - using initial guesses')
            params = [A_guess, peak_x_guess, gamma_guess, offset_guess]

        return {
            "A": float(params[0]),
            "peak_x": float(params[1]),
            "gamma": np.abs(float(params[2])),
            "offset": float(params[3])
        }

    def batch_lorentzian_analysis(self, sequences):
        """Analyze a batch of sequences with Lorentzian fitting."""
        n_sequences = len(sequences)

        fitted_As = np.zeros(n_sequences)
        fitted_peak_xs = np.zeros(n_sequences)
        fitted_gammas = np.zeros(n_sequences)
        fitted_offsets = np.zeros(n_sequences)

        fitted_lorentzian_curves = []
        fitted_lorentzian_params = []

        for i in range(n_sequences):
            sequence = sequences[i][0]
            freq_range = sequences[i][1]

            res = self.single_lorentzian_fit(freq_range, sequence)
            A = res['A']
            peak_x = res['peak_x']
            gamma = res['gamma']
            offset = res['offset']

            fitted_lorentzian_curve = self.lorentzian(freq_range, A, peak_x, gamma, offset)

            fitted_lorentzian_params.append([A, peak_x, gamma, offset])
            fitted_lorentzian_curves.append([fitted_lorentzian_curve, freq_range])

            fitted_As[i] = A
            fitted_peak_xs[i] = peak_x
            fitted_gammas[i] = gamma
            fitted_offsets[i] = offset

        return {
            "lorentzian_params": fitted_lorentzian_params,
            "lorentzian_curves": fitted_lorentzian_curves,
            "mean_As": np.mean(fitted_As),
            "std_As": np.std(fitted_As),
            "mean_peak_xs": np.mean(fitted_peak_xs),
            "std_peak_xs": np.std(fitted_peak_xs),
            "mean_gammas": np.mean(fitted_gammas),
            "std_gammas": np.std(fitted_gammas),
            "mean_offsets": np.mean(fitted_offsets),
            "std_offsets": np.std(fitted_offsets),
        }

    def batch_sequences_diff_analysis(self, sequences):
        """Analyze differences in sequence values."""
        n_sequences = len(sequences)

        mean_diffs = np.zeros(n_sequences)
        std_diffs = np.zeros(n_sequences)

        for i in range(n_sequences):
            sequence = sequences[i][0]
            diffs = np.diff(sequence)
            if diffs.size == 0:
                mean_diffs[i] = 0.0
                std_diffs[i] = 0.0
            else:
                mean_diffs[i] = np.mean(diffs)
                std_diffs[i] = np.std(diffs)

        return {
            "mean_mean_diffs": np.mean(mean_diffs),
            "std_mean_diffs": np.std(mean_diffs),
            "mean_std_diffs": np.mean(std_diffs),
            "std_std_diffs": np.std(std_diffs),
        }

    def analyze(self, title="Real Data", plot=True, num_sample_plots=2):
        """Analyze real data and extract statistics."""
        n_sequences = self.data.shape[1]

        pre_lor_sequences = []
        lorentzian_sequences = []
        post_lor_sequences = []
        min_distances = []

        for i in tqdm(range(n_sequences), desc=f"Analyzing {title} sequences"):
            sequence = self.data[:, i].flatten()
            freq_range = self.frequency_axis_mhz.flatten()

            peak_idx = np.argmax(sequence)
            pre_peak = sequence[:peak_idx]
            post_peak = sequence[peak_idx:]

            # Find minima for segmentation
            if len(pre_peak) > 3:
                left_minima = argrelextrema(pre_peak, np.less, order=3)[0]
                min1_idx = left_minima[-1] if len(left_minima) > 0 else 0
            else:
                min1_idx = 0

            if len(post_peak) > 3:
                right_minima = argrelextrema(post_peak, np.less, order=3)[0]
                min2_idx = peak_idx + (right_minima[0] if len(right_minima) > 0 else len(post_peak)-1)
            else:
                min2_idx = len(sequence)-1

            min_distances.append(freq_range[peak_idx] - freq_range[min1_idx])
            min_distances.append(freq_range[min2_idx] - freq_range[peak_idx])

            # Segment the sequence
            pre_lor_sequence = sequence[:min1_idx][::-1]
            pre_lor_freqs = freq_range[:min1_idx][::-1]

            lorentzian_sequence = sequence[min1_idx:min2_idx]
            lorentzian_freqs = freq_range[min1_idx:min2_idx]

            post_lor_sequence = sequence[min2_idx:]
            post_lor_freqs = freq_range[min2_idx:]

            pre_lor_sequences.append([pre_lor_sequence, pre_lor_freqs])
            lorentzian_sequences.append([lorentzian_sequence, lorentzian_freqs])
            post_lor_sequences.append([post_lor_sequence, post_lor_freqs])

        # Analyze each segment
        pre_lor_stats = self.batch_sequences_diff_analysis(pre_lor_sequences)
        lorentzian_results = self.batch_lorentzian_analysis(lorentzian_sequences)
        post_lor_stats = self.batch_sequences_diff_analysis(post_lor_sequences)

        # Extract peak frequencies and FWHMs
        peak_freqs = [params[1] for params in lorentzian_results["lorentzian_params"]]
        fwhms = [2*params[2] for params in lorentzian_results["lorentzian_params"]]

        # Plotting
        if plot:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.hist(peak_freqs, bins=20, alpha=0.7)
            plt.title("Peak Frequency Distribution")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Count")

            plt.subplot(1, 3, 2)
            plt.hist(fwhms, bins=20, alpha=0.7)
            plt.title("FWHM Distribution")
            plt.xlabel("FWHM (MHz)")
            plt.ylabel("Count")

            plt.suptitle(f"{title} - Statistical Analysis")
            plt.tight_layout()
            plt.show()

            # Sample individual sequence plots
            sample_indices = random.sample(range(n_sequences), min(num_sample_plots, n_sequences))

            for idx in sample_indices:
                plt.figure(figsize=(12, 6))

                sequence = self.data[:, idx].flatten()
                freq_range = self.frequency_axis_mhz.flatten()

                plt.plot(freq_range, sequence, 'b-', label='Original Data', alpha=0.7)

                pre_seq, pre_freq = pre_lor_sequences[idx]
                lor_seq, lor_freq = lorentzian_sequences[idx]
                post_seq, post_freq = post_lor_sequences[idx]

                plt.plot(pre_freq, pre_seq, 'g-', label='Pre-Lorentzian', linewidth=2)
                plt.plot(lor_freq, lor_seq, 'r-', label='Lorentzian', linewidth=2)
                plt.plot(post_freq, post_seq, 'm-', label='Post-Lorentzian', linewidth=2)

                lorentzian_curve = lorentzian_results["lorentzian_curves"][idx]
                plt.plot(lorentzian_curve[1], lorentzian_curve[0], 'k--', 
                        label='Fitted Lorentzian', linewidth=2)

                params = lorentzian_results["lorentzian_params"][idx]
                plt.axvline(params[1], color='k', linestyle=':', alpha=0.5)

                plt.title(f"{title} - Sequence {idx}\nPeak: {params[1]:.2f} MHz, FWHM: {2*params[2]:.2f} MHz")
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        self.analyze_results = {
            "pre_lor_statistics": pre_lor_stats,
            "lorentzian_analysis_results": lorentzian_results,
            "post_lor_statistics": post_lor_stats,
            'min_distances_mean': np.mean(min_distances),
            'min_distances_std': np.std(min_distances),
            "peak_frequencies": peak_freqs,
            "fwhms": fwhms,
        }

    def gen_synth(self, n=1000, noise_std=0.01):
        """Generate synthetic data based on real data statistics."""
        if not self.analyze_results:
            raise ValueError("Analyze results are missing. Run analyze() before gen_synth().")

        statistics = self.analyze_results
        pre_lor_stats = statistics['pre_lor_statistics']
        lorentzian_stats = statistics['lorentzian_analysis_results']
        post_lor_stats = statistics['post_lor_statistics']

        pre_mean_mean_diffs = pre_lor_stats['mean_mean_diffs']
        pre_std_mean_diffs = pre_lor_stats['std_mean_diffs']
        pre_mean_std_diffs = pre_lor_stats['mean_std_diffs']
        pre_std_std_diffs = pre_lor_stats['std_std_diffs']

        post_mean_mean_diffs = post_lor_stats['mean_mean_diffs']
        post_std_mean_diffs = post_lor_stats['std_mean_diffs']
        post_mean_std_diffs = post_lor_stats['mean_std_diffs']
        post_std_std_diffs = post_lor_stats['std_std_diffs']

        mean_As = lorentzian_stats['mean_As']
        std_As = lorentzian_stats['std_As']
        mean_peak_xs = lorentzian_stats['mean_peak_xs']
        std_peak_xs = lorentzian_stats['std_peak_xs']
        mean_gammas = lorentzian_stats['mean_gammas']
        std_gammas = lorentzian_stats['std_gammas']
        mean_offsets = lorentzian_stats['mean_offsets']
        std_offsets = lorentzian_stats['std_offsets']

        min_distances_mean = statistics['min_distances_mean']
        min_distances_std = statistics['min_distances_std']

        synthetic_sequences = np.zeros((68, n))
        synthetic_sequences_noisy = np.zeros((68, n))
        peak_xs = np.zeros(n)
        fwhms = np.zeros(n)

        for i in range(n):
            freq_range = self.frequency_axis_mhz.flatten()

            gen_A = np.random.normal(mean_As, std_As)
            gen_peak_x = np.random.normal(mean_peak_xs, std_peak_xs)
            gen_gamma = np.random.normal(mean_gammas, std_gammas)
            gen_offset = np.random.normal(mean_offsets, std_offsets)

            peak_xs[i] = gen_peak_x
            fwhms[i] = 2*gen_gamma

            min1_distance = np.random.normal(min_distances_mean, min_distances_std)
            min2_distance = np.random.normal(min_distances_mean, min_distances_std)
            min1_x = gen_peak_x - min1_distance
            min2_x = gen_peak_x + min2_distance

            min1_idx, min2_idx = self.freq_to_idx(np.array([min1_x, min2_x]))
            max_idx = self.frequency_axis_mhz.size - 1
            min1_idx = int(np.clip(min1_idx, 0, max_idx - 1))
            min2_idx = int(np.clip(min2_idx, min1_idx + 1, max_idx))

            lor_freq_range = freq_range[min1_idx:min2_idx]
            lorentzian_seq = self.lorentzian(lor_freq_range, gen_A, gen_peak_x, gen_gamma, gen_offset)

            # Generate pre-Lorentzian sequence
            pre_diff_mean = np.random.normal(pre_mean_mean_diffs, pre_std_mean_diffs)
            pre_diff_std = np.abs(np.random.normal(pre_mean_std_diffs, pre_std_std_diffs))
            pre_seq = [lorentzian_seq[0]]
            for j in range(min1_idx-1):
                diff = np.random.normal(pre_diff_mean, pre_diff_std)
                pre_seq.append(pre_seq[-1]+diff)
            pre_seq = pre_seq[::-1]
            synthetic_sequences[:len(pre_seq), i] = pre_seq

            # Add Lorentzian sequence
            synthetic_sequences[len(pre_seq):len(pre_seq)+len(lorentzian_seq), i] = lorentzian_seq

            # Generate post-Lorentzian sequence
            post_diff_mean = np.random.normal(post_mean_mean_diffs, post_std_mean_diffs)
            post_diff_std = np.abs(np.random.normal(post_mean_std_diffs, post_std_std_diffs))
            post_seq = [lorentzian_seq[-1]]
            for j in range(68 - min2_idx - 1):
                diff = np.random.normal(post_diff_mean, post_diff_std)
                post_seq.append(post_seq[-1]+diff)
            synthetic_sequences[min2_idx:, i] = post_seq[:68-min2_idx]

            # Add noise
            seq = synthetic_sequences[:, i]
            noise_std_high = noise_std*gen_A*2
            noise_std_med = noise_std*gen_A
            noise_std_low = noise_std*gen_A*0.5
            random_noise_std = np.random.choice([noise_std_high, noise_std_med, noise_std_low])
            synthetic_sequences_noisy[:, i] = seq + np.random.normal(0, random_noise_std, size=seq.shape[0])

        self.synthetic_data = {
            "synthetic_sequences": synthetic_sequences_noisy,
            "peak_xs": peak_xs,
            "fwhms": fwhms
        }

    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess synthetic data for model training."""
        if not self.synthetic_data:
            raise ValueError("Synthetic data is missing. Run gen_synth() before preprocess_data().")

        sequences = self.synthetic_data["synthetic_sequences"]
        peaks = self.synthetic_data["peak_xs"]
        fwhms = self.synthetic_data["fwhms"]
        
        print("Preprocessing data...")
        n_sequences = sequences.shape[1]
        normalized_sequences = np.zeros_like(sequences)
        
        for i in tqdm(range(n_sequences), desc="Normalizing sequences"):
            scaler = MinMaxScaler()
            normalized_sequences[:, i] = scaler.fit_transform(
                sequences[:, i].reshape(-1, 1)
            ).flatten()

        X = normalized_sequences.T.reshape((-1, 68, 1))

        peaks_scaler = StandardScaler()
        normalized_peaks = peaks_scaler.fit_transform(peaks.reshape(-1, 1)).flatten()

        fwhms_scaler = StandardScaler()
        normalized_fwhms = fwhms_scaler.fit_transform(fwhms.reshape(-1, 1)).flatten()

        X_train, X_test, y_peak_train, y_peak_test, y_fwhm_train, y_fwhm_test = train_test_split(
            X, normalized_peaks, normalized_fwhms, test_size=test_size, random_state=random_state
        )

        y_train = {
            'peak_output': y_peak_train,
            'fwhm_output': y_fwhm_train
        }
        y_test = {
            'peak_output': y_peak_test,
            'fwhm_output': y_fwhm_test
        }

        # Visualizations
        plt.figure(figsize=(18, 12))
        plot_cols = 2
        plot_rows = 3
        n_samples = plot_cols*plot_rows

        for i in range(n_samples):
            sample_idx = np.random.randint(0, X.shape[0])
            normalized_sequence = X[sample_idx].flatten()
            normalized_peak = normalized_peaks[sample_idx]
            normalized_fwhm = normalized_fwhms[sample_idx]

            peak = peaks_scaler.inverse_transform(normalized_peak.reshape((-1, 1))).flatten()[0]
            fwhm = fwhms_scaler.inverse_transform(normalized_fwhm.reshape((-1, 1))).flatten()[0]

            plt.subplot(plot_rows, plot_cols, i+1)
            plt.plot(self.frequency_axis_mhz, normalized_sequence, label='Normalized')
            plt.axvline(peak, color='r', linestyle='--', label='Peak')
            plt.title(f'Sample Sequence (Peak: {peak:.2f}, FWHM: {fwhm:.2f})')
            plt.legend()

        plt.tight_layout()
        plt.show()

        print("Preprocessing complete. Train shape:", X_train.shape)

        print("\nSaving scalers...")
        with open(self.scalers_dir / 'fwhms_scaler.pkl', 'wb') as f:
            pickle.dump(fwhms_scaler, f)
        with open(self.scalers_dir / 'peaks_scaler.pkl', 'wb') as f:
            pickle.dump(peaks_scaler, f)

        return X_train, X_test, y_train, y_test

    def compile_model(self, initial_lr=0.001):
        """Compile the model with dual outputs."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss={
                'peak_output': 'mse',
                'fwhm_output': 'mse'
            },
            loss_weights={
                'peak_output': 0.5,
                'fwhm_output': 0.5
            },
        )

        self.callbacks = self._build_callbacks()
        print("Model compiled with callbacks:", [cb.__class__.__name__ for cb in self.callbacks])

    def model_architecture(self):
        """Create dual-output CNN model for peak and FWHM prediction."""
        print("Creating dual-output CNN model...")
        inputs = Input(shape=(68, 1), name='input')

        x = Conv1D(filters=128, kernel_size=8, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=3)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=128, kernel_size=8, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=3)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Peak frequency prediction branch
        peak_branch = Dense(128, activation='relu')(x)
        peak_branch = Dense(64, activation='relu')(peak_branch)
        peak_branch = BatchNormalization()(peak_branch)
        peak_output = Dense(1, name='peak_output')(peak_branch)

        # FWHM prediction branch
        fwhm_branch = Dense(128, activation='relu')(x)
        fwhm_branch = Dense(64, activation='relu')(fwhm_branch)
        fwhm_branch = BatchNormalization()(fwhm_branch)
        fwhm_output = Dense(1, name='fwhm_output')(fwhm_branch)

        model = Model(inputs=inputs, outputs=[peak_output, fwhm_output])

        print("Model architecture created successfully!")
        model.summary()
        self.model = model

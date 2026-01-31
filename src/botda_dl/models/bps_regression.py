"""
BPS (Brillouin Phase Spectrum) Regression analysis module.
Implements regression-based peak frequency estimation.
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.regularizers import l2

from ..core import BaseAnalyzer


class BPSRegression(BaseAnalyzer):
    """
    BPS Regression-based analyzer for peak frequency estimation.
    Analyzes Brillouin Phase Spectrum data.
    """

    def calculate_sequence_stats(self, sequences_list):
        """Calculate statistical properties of sequence differences."""
        n_sequences = len(sequences_list)

        mean_diffs = np.zeros((n_sequences, 1))
        std_diffs = np.zeros((n_sequences, 1))

        for i in tqdm(range(n_sequences), desc="Calculating sequence stats"):
            sequence = np.array(sequences_list[i])
            diffs = np.diff(sequence)
            if diffs.size == 0:
                mean_diffs[i] = 0.0
                std_diffs[i] = 0.0
            else:
                mean_diffs[i] = np.mean(diffs)
                std_diffs[i] = np.std(diffs)

        return {
            'mean_mean_diffs': np.mean(mean_diffs),
            'std_mean_diffs': np.std(mean_diffs),
            'mean_std_diffs': np.mean(std_diffs),
            'std_std_diffs': np.std(std_diffs)
        }

    def analyze(self, title="Real Data", plot=True):
        """Analyze real BPS data and extract statistics."""
        print("Starting data analysis...")
        n_sequences = self.data.shape[1]

        post_peak_segments = []
        pre_peak_segments = []

        peak_xs = []
        peak_vals = []
        min_vals = []

        for i in tqdm(range(n_sequences), desc="Segmenting sequences"):
            sequence = self.data[:, i]

            min_idx = np.argmin(sequence)
            min_x = self.idx_to_freq(min_idx)
            min_val = sequence[min_idx]

            max_idx = np.argmax(sequence)
            max_x = self.idx_to_freq(max_idx)
            max_val = sequence[max_idx]

            peak_x = (max_x + min_x) / 2
            peak_xs.append(peak_x)
            min_vals.append(min_val)
            peak_vals.append(max_val - min_val)

            pre_peak_segments.append(sequence[:min_idx+1][::-1])
            post_peak_segments.append(sequence[max_idx:])

        pre_peak_stats = self.calculate_sequence_stats(pre_peak_segments)
        post_peak_stats = self.calculate_sequence_stats(post_peak_segments)

        peak_stats = {
            'mean_peak_xs': np.mean(peak_xs),
            'std_peak_xs': np.std(peak_xs),
            'mean_peak_vals': np.mean(peak_vals),
            'std_peak_vals': np.std(peak_vals),
            'mean_min_vals': np.mean(min_vals),
            'std_min_vals': np.std(min_vals),
        }

        if plot:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.hist(peak_xs, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Peak Frequency Distribution')
            plt.xlabel('Peak Frequency (MHz)')
            plt.ylabel('Count')

            plt.subplot(2, 3, 2)
            plt.hist(peak_vals, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Peak Value Range Distribution')
            plt.xlabel('Peak Value Range')
            plt.ylabel('Count')

            plt.subplot(2, 3, 3)
            plt.scatter(peak_xs, peak_vals, alpha=0.5)
            plt.title('Peak Frequency vs Value Range')
            plt.xlabel('Peak Frequency (MHz)')
            plt.ylabel('Peak Value Range')

            plt.subplot(2, 3, 4)
            for i in range(min(5, n_sequences)):
                plt.plot(self.frequency_axis_mhz, self.data[:, i], alpha=0.5)
                plt.axvline(x=peak_xs[i], color='r', linestyle='--', alpha=0.3)
            plt.title('Sample Sequences with Peaks')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')

            plt.subplot(2, 3, 5)
            plt.hist(min_vals, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Minimum Value Distribution')
            plt.xlabel('Minimum Value')
            plt.ylabel('Count')

            plt.suptitle(f"{title} - Statistical Analysis")
            plt.tight_layout()
            plt.show()

        self.analyze_results = {
            "pre_peak_statistics": pre_peak_stats,
            "post_peak_statistics": post_peak_stats,
            "peak_statistics": peak_stats,
            "peak_frequencies": peak_xs,
        }

    def gen_synth(self, n=1000, noise_std=0.01):
        """Generate synthetic BPS data based on real data statistics."""
        if not self.analyze_results:
            raise ValueError("Analyze results are missing. Run analyze() before gen_synth().")

        statistics = self.analyze_results
        pre_stats = statistics['pre_peak_statistics']
        post_stats = statistics['post_peak_statistics']
        peak_stats = statistics['peak_statistics']

        freq_range = self.frequency_axis_mhz.flatten()

        synthetic_sequences = np.zeros((68, n))
        synthetic_sequences_noisy = np.zeros((68, n))
        peak_xs = np.zeros(n)

        for i in range(n):
            gen_peak_x = np.random.normal(peak_stats['mean_peak_xs'], peak_stats['std_peak_xs'])
            gen_peak_val = np.random.normal(peak_stats['mean_peak_vals'], peak_stats['std_peak_vals'])
            gen_min_val = np.random.normal(peak_stats['mean_min_vals'], peak_stats['std_min_vals'])

            peak_xs[i] = gen_peak_x
            peak_idx = self.freq_to_idx(np.array([gen_peak_x]))[0]

            # Generate pre-peak segment
            pre_diff_mean = np.random.normal(pre_stats['mean_mean_diffs'], pre_stats['std_mean_diffs'])
            pre_diff_std = np.abs(np.random.normal(pre_stats['mean_std_diffs'], pre_stats['std_std_diffs']))

            pre_seq = [gen_min_val]
            for j in range(peak_idx):
                diff = np.random.normal(pre_diff_mean, pre_diff_std)
                pre_seq.append(pre_seq[-1] + diff)

            pre_seq = pre_seq[::-1]
            synthetic_sequences[:len(pre_seq), i] = pre_seq[:peak_idx+1]

            # Set peak value
            synthetic_sequences[peak_idx, i] = gen_min_val + gen_peak_val

            # Generate post-peak segment
            post_diff_mean = np.random.normal(post_stats['mean_mean_diffs'], post_stats['std_mean_diffs'])
            post_diff_std = np.abs(np.random.normal(post_stats['mean_std_diffs'], post_stats['std_std_diffs']))

            post_seq = [gen_min_val + gen_peak_val]
            for j in range(68 - peak_idx - 1):
                diff = np.random.normal(post_diff_mean, post_diff_std)
                post_seq.append(post_seq[-1] + diff)

            synthetic_sequences[peak_idx:, i] = post_seq[:68-peak_idx]

            # Add noise
            seq = synthetic_sequences[:, i]
            random_noise_std = noise_std * (gen_peak_val if gen_peak_val > 0 else 1)
            synthetic_sequences_noisy[:, i] = seq + np.random.normal(0, random_noise_std, size=seq.shape[0])

        self.synthetic_data = {
            "synthetic_sequences": synthetic_sequences_noisy,
            "peak_xs": peak_xs,
        }

    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess synthetic data for model training."""
        if not self.synthetic_data:
            raise ValueError("Synthetic data is missing. Run gen_synth() before preprocess_data().")

        print("Starting data preprocessing...")

        sequences = self.synthetic_data["synthetic_sequences"]
        peak_xs = self.synthetic_data["peak_xs"]

        normalized_sequences = np.zeros_like(sequences)

        for i in tqdm(range(sequences.shape[1]), desc="Normalizing sequences"):
            sequence = sequences[:, i]
            scaler = MinMaxScaler()
            normalized_sequences[:, i] = scaler.fit_transform(sequence.reshape(-1, 1)).flatten()

        X = normalized_sequences.T.reshape((-1, 68, 1))

        peaks_scaler = StandardScaler()
        normalized_peaks = peaks_scaler.fit_transform(peak_xs.reshape(-1, 1)).flatten()
        
        print(f"Mean: {np.mean(normalized_peaks):.4f}, Std: {np.std(normalized_peaks):.4f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, normalized_peaks, test_size=test_size, random_state=random_state
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Visualizations
        plt.figure(figsize=(18, 12))

        sample_idx = np.random.randint(0, sequences.shape[1])
        plt.subplot(2, 3, 1)
        plt.plot(self.frequency_axis_mhz, normalized_sequences[:, sample_idx], label='Normalized')
        plt.axvline(x=peak_xs[sample_idx], color='r', linestyle='--', label='Peak')
        plt.title(f'Sample Sequence (Index {sample_idx})')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Normalized Value')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.hist(peak_xs, bins=68)
        plt.title('Distribution of Peak Frequencies')
        plt.xlabel('Peak Frequencies')
        plt.ylabel('Count')

        plt.subplot(2, 3, 4)
        for i in range(min(5, sequences.shape[1])):
            plt.plot(self.frequency_axis_mhz, normalized_sequences[:, i], alpha=0.5)
            plt.axvline(x=peak_xs[i], color='r', linestyle='--', alpha=0.3)
        plt.title('Multiple Normalized Sequences with Peaks')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Normalized Value')

        plt.subplot(2, 3, 5)
        plt.bar(['Train', 'Test'], [len(X_train), len(X_test)])
        plt.title('Train-Test Split')
        plt.ylabel('Number of Samples')

        plt.tight_layout()
        plt.show()

        print("\nSaving scaler for future use...")
        with open(self.scalers_dir / 'peaks_scaler.pkl', 'wb') as f:
            pickle.dump(peaks_scaler, f)

        print("Preprocessing completed.")

        return X_train, X_test, y_train, y_test

    def compile_model(self, initial_lr=0.001):
        """Compile the model for single output (peak frequency)."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss={'peak_output': 'mse'},
            metrics={'peak_output': 'mae'}
        )

        self.callbacks = self._build_callbacks()
        print("Model compiled with callbacks:", [cb.__class__.__name__ for cb in self.callbacks])

    def model_architecture(self):
        """Create single-output CNN model for peak frequency prediction."""
        print("Creating CNN model for peak frequency estimation...")
        inputs = Input(shape=(68, 1), name='input')

        x = Conv1D(filters=64, kernel_size=8, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=3)(x)
        x = Dropout(0.2)(x)

        x = Conv1D(filters=128, kernel_size=8, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=3)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Peak frequency output
        peak_output = Dense(1, name='peak_output')(x)

        model = Model(inputs=inputs, outputs=peak_output)

        print("Model architecture created successfully!")
        model.summary()
        self.model = model

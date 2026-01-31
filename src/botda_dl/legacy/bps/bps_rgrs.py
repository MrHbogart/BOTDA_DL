from .bps import BPS
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2


class BPS_RGRS(BPS):
    def calculate_sequence_stats(self, sequences_list):
        n_sequences = len(sequences_list)

        mean_diffs = np.zeros((n_sequences, 1))
        std_diffs = np.zeros((n_sequences, 1))

        for i in tqdm(range(n_sequences), desc="Calculating sequence stats"):
            sequence = np.array(sequences_list[i])

            diffs = np.diff(sequence)
            mean_diffs[i] = np.mean(diffs)
            std_diffs[i] = np.std(diffs)

        mean_mean_diffs = np.mean(mean_diffs)
        std_mean_diffs = np.std(mean_diffs)

        mean_std_diffs = np.mean(std_diffs)
        std_std_diffs = np.std(std_diffs)

        return {'mean_mean_diffs': mean_mean_diffs,
                    'std_mean_diffs': std_mean_diffs,
                    'mean_std_diffs': mean_std_diffs,
                    'std_std_diffs': std_std_diffs}
    
    def analyze(self, title="Real Data", plot=True):
        print("Starting data analysis...")
        n_sequences = self.data.shape[1]

        post_peak_segments = []
        pre_peak_segments = []

        bfs_xs = []
        peak_vals = []
        min_xs = []
        min_vals = []

        for i in tqdm(range(n_sequences), desc="Segmenting sequences"):
            sequence = self.data[:, i]

            min_idx = np.argmin(sequence)
            min_x = self.idx_to_freq(min_idx)
            min_val = sequence[min_idx]

            max_idx = np.argmax(sequence)
            max_x = self.idx_to_freq(max_idx)
            max_val = sequence[max_idx]

            bfs_x = (max_x+min_x)/2
            bfs_xs.append(bfs_x)

            min_vals.append(min_val)

            peak_vals.append(max_val-min_val)

            pre_peak_segments.append(sequence[:min_idx+1][::-1])
            post_peak_segments.append(sequence[max_idx:])

        pre_peak_stats = self.calculate_sequence_stats(pre_peak_segments)
        post_peak_stats = self.calculate_sequence_stats(post_peak_segments)

        peak_stats = {
            'mean_bfs_xs': np.mean(bfs_xs),
            'std_bfs_xs': np.std(bfs_xs),

            'mean_peak_vals': np.mean(peak_vals),
            'std_peak_vals': np.std(peak_vals),

            'mean_min_vals': np.mean(min_vals),
            'std_min_vals': np.std(min_vals),
        }

        results = {'peak_stats': peak_stats,
                'pre_peak_stats': pre_peak_stats,
                'post_peak_stats': post_peak_stats}


        # Visualizations
        plt.figure(figsize=(14, 8))

        # Example sequences
        plt.subplot(2, 2, 1)
        for i in range(min(5, self.data.shape[1])):
            plt.plot(self.frequency_axis_mhz, self.data[:, i], alpha=0.7)
        plt.title('Example Real Sequences')
        plt.xlabel('Frequency')
        plt.ylabel('Value')

        # BFS distribution
        plt.subplot(2, 2, 2)
        plt.hist(bfs_xs, bins=20)
        plt.title('Distribution of BFSs')
        plt.xlabel('BFS')
        plt.ylabel('Count')

        # Peak values distribution
        plt.subplot(2, 2, 3)
        plt.hist(peak_vals, bins=20)
        plt.title('Distribution of Peak Values')
        plt.xlabel('Peak Value')
        plt.ylabel('Count')

        # Additional plot: Average Real sequence
        plt.subplot(2, 2, 4)
        avg_synth = np.mean(self.data, axis=1)
        std_synth = np.std(self.data, axis=1)
        plt.plot(self.frequency_axis_mhz, avg_synth, label='Average')
        plt.fill_between(self.frequency_axis_mhz, avg_synth - std_synth, avg_synth + std_synth, alpha=0.2)
        plt.title('Average Real Sequence ± Std')
        plt.xlabel('Frequency')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.show()


        print("Data analysis completed.")

        self.analyze_results =  {
            "real_data_stats": results,
            "bfs_xs": bfs_xs,
        }   
    
    def gen_synth(self, n=1000, noise_std=0.01):

        print(f"Generating {n} synthetic sequences...")
        synthetic_sequences = np.zeros((68, n))

        bfs_xs = np.zeros((n,))
        peak_vals = np.zeros((n,))

        real_data_stats = self.analyze_results['real_data_stats']

        peak_stats = real_data_stats['peak_stats']
        pre_stats = real_data_stats['pre_peak_stats']
        post_stats = real_data_stats['post_peak_stats']

        for i in tqdm(range(n), desc="Generating sequences"):

            min_val = np.random.normal(peak_stats['mean_min_vals'], peak_stats['std_min_vals'])

            bfs_x = np.random.normal(peak_stats['mean_bfs_xs'], 3*peak_stats['std_bfs_xs'])
            peak_val = np.random.normal(peak_stats['mean_peak_vals'], peak_stats['std_peak_vals'])

            peak_vals[i] = peak_val

            peak_len = np.random.randint(1, 3)

            # Pre-peak sequence
            seq = [min_val]
            mean_diff = np.random.normal(pre_stats['mean_mean_diffs'], pre_stats['std_mean_diffs'])
            std_diff = max(pre_stats['mean_std_diffs'], np.random.normal(pre_stats['mean_std_diffs'], pre_stats['std_std_diffs']))
            bfs_idx = self.freq_to_idx(np.array([bfs_x]))[0]
            for _ in range(bfs_idx-peak_len//2):
                diff = np.random.normal(mean_diff, std_diff)
                next_val = max(min_val, seq[-1]+diff +1e-2)
                seq.append(next_val)

            seq = seq[::-1]

            # Add peak
            for j in range(peak_len):
                seq.append(seq[-1] + peak_val/peak_len)

            # Post-peak sequence
            remaining_length = 68 - len(seq)
            mean_diff = np.random.normal(post_stats['mean_mean_diffs'], post_stats['std_mean_diffs'])
            std_diff = max(post_stats['mean_std_diffs'], np.random.normal(post_stats['mean_std_diffs'], post_stats['std_std_diffs']))

            for _ in range(remaining_length):
                diff = np.random.normal(mean_diff, std_diff)
                next_val = max(min_val, seq[-1] + diff + 1e-2)
                seq.append(next_val)

            seq = np.array(seq)

            bfs_x = (self.idx_to_freq(np.argmax(seq))+self.idx_to_freq(np.argmin(seq)))/2
            bfs_xs[i] = bfs_x
            seq = seq + np.random.normal(0, noise_std*(np.max(seq)-np.min(seq)), size=seq.shape)
            synthetic_sequences[:, i] = seq

        # Visualizations
        plt.figure(figsize=(14, 8))

        # Example sequences
        plt.subplot(2, 2, 1)
        for i in range(min(5, synthetic_sequences.shape[1])):
            plt.plot(self.frequency_axis_mhz, synthetic_sequences[:, i], alpha=0.7)
        plt.title('Example Synthetic Sequences')
        plt.xlabel('Index')
        plt.ylabel('Value')

        # BFS distribution
        plt.subplot(2, 2, 2)
        plt.hist(bfs_xs, bins=20)
        plt.title('Distribution of BFSs')
        plt.xlabel('BFS')
        plt.ylabel('Count')

        # Peak values distribution
        plt.subplot(2, 2, 3)
        plt.hist(peak_vals, bins=20)
        plt.title('Distribution of Peak Values')
        plt.xlabel('Peak Value')
        plt.ylabel('Count')

        # Additional plot: Average synthetic sequence
        plt.subplot(2, 2, 4)
        avg_synth = np.mean(synthetic_sequences, axis=1)
        std_synth = np.std(synthetic_sequences, axis=1)
        plt.plot(self.frequency_axis_mhz, avg_synth, label='Average')
        plt.fill_between(self.frequency_axis_mhz, avg_synth - std_synth, avg_synth + std_synth, alpha=0.2)
        plt.title('Average Synthetic Sequence ± Std')
        plt.xlabel('Frequency')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("Synthetic data generation completed.")
        self.synthetic_data = {
            "synthetic_sequences": synthetic_sequences,
            "bfs_xs": bfs_xs,
        }
    
    def model_architecture(self):
        print("Creating single-output CNN model...")
        # Input layer
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

        # BFS prediction branch (regression)
        peak_branch = Dense(128, activation='relu')(x)
        peak_branch = Dense(64, activation='relu')(peak_branch)
        peak_branch = BatchNormalization()(peak_branch)
        bfs_output = Dense(1, name='bfs_output')(peak_branch)

        model = Model(inputs=inputs, outputs=bfs_output)
        print("Model architecture created successfully!")
        model.summary()
        self.model = model

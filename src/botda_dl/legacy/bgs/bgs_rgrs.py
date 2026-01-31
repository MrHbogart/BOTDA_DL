from .bgs import BGS
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
import seaborn as sns


class BGS_RGRS(BGS):
    def lorentzian(self, x, A, bfs_x, gamma, offset):
        return offset + (A * gamma**2) / ((x - bfs_x)**2 + gamma**2)
    
    def single_lorentzian_freq_gain(self, x, y):
        try:
            A_guess = np.max(y)
            bfs_x_guess = x[x.shape[0]//2]
            gamma_guess = x[-x.shape[0]//4] - x[x.shape[0]//4]
            offset_guess = np.min(y)
            params, _ = curve_fit(self.lorentzian, x, y,
                                p0=[A_guess, bfs_x_guess , gamma_guess, offset_guess],
                                maxfev=5000)

        except RuntimeError:
            print('Lorentzian fit failed - using initial guesses')
            params = [A_guess, bfs_x_guess , gamma_guess, offset_guess]

        return {"A": float(params[0]),
                "bfs_x": float(params[1]),
                "gamma": np.abs(float(params[2])),
                "offset": float(params[3])}

    def batch_lorentzian_analysis(self, sequences):
        n_sequences = len(sequences)

        fitted_As = np.zeros((n_sequences))
        fitted_bfs_xs = np.zeros((n_sequences))
        fitted_gammas = np.zeros((n_sequences))
        fitted_offsets = np.zeros((n_sequences))

        fitted_lorentzian_curves = []
        fitted_lorentzian_params = []

        for i in range(n_sequences):
            sequence = sequences[i][0]
            freq_range = sequences[i][1]

            res = self.single_lorentzian_freq_gain(freq_range, sequence)
            A = res['A']
            bfs_x = res['bfs_x']
            gamma = res['gamma']
            offset = res['offset']

            fitted_lorentzian_curve = self.lorentzian(freq_range, A, bfs_x, gamma, offset)

            fitted_lorentzian_params.append([A, bfs_x, gamma, offset])
            fitted_lorentzian_curves.append([fitted_lorentzian_curve, freq_range])

            fitted_As[i] = A
            fitted_bfs_xs[i] = bfs_x
            fitted_gammas[i] = gamma
            fitted_offsets[i] = offset

        result = {
            "lorentzian_params": fitted_lorentzian_params,
            "lorentzian_curves": fitted_lorentzian_curves,

            "mean_As": np.mean(fitted_As),
            "std_As": np.std(fitted_As),

            "mean_bfs_xs": np.mean(fitted_bfs_xs),
            "std_bfs_xs": np.std(fitted_bfs_xs),

            "mean_gammas": np.mean(fitted_gammas),
            "std_gammas": np.std(fitted_gammas),

            "mean_offsets": np.mean(fitted_offsets),
            "std_offsets": np.std(fitted_offsets),
        }

        return result

    def batch_sequences_diff_analysis(self, sequences):
        n_sequences = len(sequences)

        mean_diffs = np.zeros((n_sequences))
        std_diffs = np.zeros((n_sequences))

        for i in range(n_sequences):
            sequence = sequences[i][0]

            diffs = np.diff(sequence)
            mean_diffs[i] = np.mean(diffs)
            std_diffs[i] = np.std(diffs)


        return {
            "mean_mean_diffs": np.mean(mean_diffs),
            "std_mean_diffs": np.std(mean_diffs),

            "mean_std_diffs": np.mean(std_diffs),
            "std_std_diffs": np.std(std_diffs),
        }

    def analyze(self, title="Real Data", plot=True, num_sample_plots=2):
        n_sequences = self.data.shape[1]

        pre_lor_sequences = []
        lorentzian_sequences = []
        post_lor_sequences = []

        min_distances = []

        for i in tqdm(range(n_sequences), desc=f"analyzing {title} sequences"):
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
        pre_lor_statistics = self.batch_sequences_diff_analysis(pre_lor_sequences)
        lorentzian_results = self.batch_lorentzian_analysis(lorentzian_sequences)
        post_lor_statistics = self.batch_sequences_diff_analysis(post_lor_sequences)

        # Extract bfs frequencies and FWHMs
        bfs_freqs = []
        fwhms = []
        for curve_params in lorentzian_results["lorentzian_params"]:
            bfs_freqs.append(curve_params[1])
            fwhms.append(2*curve_params[2])

        # Plotting section
        if plot:
            palette = sns.color_palette('colorblind', 6)
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.hist(bfs_freqs, bins=20, alpha=0.8, color=palette[0], edgecolor='black')
            plt.title("BFS Distribution", fontsize=12)
            plt.xlabel("Frequency (MHz)", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')

            plt.subplot(1, 3, 2)
            plt.hist(fwhms, bins=20, alpha=0.8, color=palette[1], edgecolor='black')
            plt.title("FWHM Distribution", fontsize=12)
            plt.xlabel("FWHM (MHz)", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')

            plt.suptitle(f"{title} - Statistical Analysis", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{self.results_dir}/{title}_stat_analysis.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_stat_analysis.png', format='png', dpi=300)
            plt.show()

            # Sample individual sequence plots
            sample_indices = random.sample(range(n_sequences), min(num_sample_plots, n_sequences))
            for idx in sample_indices:
                plt.figure(figsize=(12, 6))
                sequence = self.data[:, idx].flatten()
                freq_range = self.frequency_axis_mhz.flatten()
                plt.plot(freq_range, sequence, 'b-', label='Original Data', alpha=0.7, linewidth=2)
                pre_seq, pre_freq = pre_lor_sequences[idx]
                lor_seq, lor_freq = lorentzian_sequences[idx]
                post_seq, post_freq = post_lor_sequences[idx]
                plt.plot(pre_freq, pre_seq, color=palette[2], label='Pre-Lorentzian', linewidth=2)
                plt.plot(lor_freq, lor_seq, color=palette[3], label='Lorentzian', linewidth=2)
                plt.plot(post_freq, post_seq, color=palette[4], label='Post-Lorentzian', linewidth=2)
                lorentzian_curve = lorentzian_results["lorentzian_curves"][idx]
                plt.plot(lorentzian_curve[1], lorentzian_curve[0], 'k--', label='Fitted Lorentzian', linewidth=2)
                lorentzian_curve_params = lorentzian_results["lorentzian_params"][idx]
                plt.axvline(lorentzian_curve_params[1], color='k', linestyle=':', alpha=0.5)
                half_max = lorentzian_curve_params[0]/(2*lorentzian_curve_params[2]**2)
                plt.hlines(half_max, lorentzian_curve_params[1]-lorentzian_curve_params[2], lorentzian_curve_params[1]+lorentzian_curve_params[2],
                        colors='k', linestyles=':', linewidth=1.5)
                plt.title(f"{title} - Sequence {idx}\nPeak: {lorentzian_curve_params[1]:.2f} MHz, FWHM: {2*lorentzian_curve_params[2]:.2f} MHz", fontsize=8)
                plt.xlabel("Frequency (MHz)", fontsize=6)
                plt.ylabel("Amplitude", fontsize=6)
                plt.legend(frameon=False, loc='best')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                plt.savefig(f'{self.results_dir}/{title}_seq_{idx}.svg', format='svg')
                plt.savefig(f'{self.results_dir}/{title}_seq_{idx}.png', format='png', dpi=300)
                plt.show()

        self.analyze_results =  {
            "pre_lor_statistics": pre_lor_statistics,
            "lorentzian_analysis_results": lorentzian_results,
            "post_lor_statistics": post_lor_statistics,
            'min_distances_mean': np.mean(min_distances),
            'min_distances_std': np.std(min_distances),
            "bfs_frequencies": bfs_freqs,
            "fwhms": fwhms,
        }

    def gen_synth(self, n=1000, noise_std=0.01):

        statistics = self.analyze_results
        pre_lor_statistics = statistics['pre_lor_statistics']
        lorentzian_statistics = statistics['lorentzian_analysis_results']
        post_lor_statistics = statistics['post_lor_statistics']

        pre_mean_mean_diffs = pre_lor_statistics['mean_mean_diffs']
        pre_std_mean_diffs = pre_lor_statistics['std_mean_diffs']
        pre_mean_std_diffs = pre_lor_statistics['mean_std_diffs']
        pre_std_std_diffs = pre_lor_statistics['std_std_diffs']

        post_mean_mean_diffs = post_lor_statistics['mean_mean_diffs']
        post_std_mean_diffs = post_lor_statistics['std_mean_diffs']
        post_mean_std_diffs = post_lor_statistics['mean_std_diffs']
        post_std_std_diffs = post_lor_statistics['std_std_diffs']

        mean_As = lorentzian_statistics['mean_As']
        std_As = lorentzian_statistics['std_As']
        mean_bfs_xs = lorentzian_statistics['mean_bfs_xs']
        std_bfs_xs = lorentzian_statistics['std_bfs_xs']
        mean_gammas = lorentzian_statistics['mean_gammas']
        std_gammas = lorentzian_statistics['std_gammas']
        mean_offsets = lorentzian_statistics['mean_offsets']
        std_offsets = lorentzian_statistics['std_offsets']

        min_distances_mean = statistics['min_distances_mean']
        min_distances_std = statistics['min_distances_std']


        synthetic_sequences = np.zeros((68, n))
        synthetic_sequences_noisy = np.zeros((68, n))
        bfs_xs = np.zeros((n))
        fwhms = np.zeros((n))
        for i in range(n):
            freq_range = self.frequency_axis_mhz.flatten()

            gen_A = np.random.normal(mean_As, std_As)
            gen_bfs_x = np.random.normal(mean_bfs_xs, std_bfs_xs)
            gen_gamma = np.random.normal(mean_gammas, std_gammas)
            gen_offset = np.random.normal(mean_offsets, std_offsets)

            bfs_xs[i] = gen_bfs_x
            fwhms[i] = 2*gen_gamma


            min1_distance = np.random.normal(min_distances_mean, min_distances_std)
            min2_distance = np.random.normal(min_distances_mean, min_distances_std)
            min1_x = gen_bfs_x - min1_distance
            min2_x = gen_bfs_x + min2_distance

            min1_idx, min2_idx = self.freq_to_idx(np.array([min1_x, min2_x]))

            lor_freq_range = freq_range[min1_idx:min2_idx]

            lorentzian_seq = self.lorentzian(lor_freq_range, gen_A, gen_bfs_x, gen_gamma, gen_offset)

            pre_diff_mean = np.random.normal(pre_mean_mean_diffs, pre_std_mean_diffs)
            pre_diff_std = np.abs(np.random.normal(pre_mean_std_diffs, pre_std_std_diffs))
            pre_seq = [lorentzian_seq[0]]
            for j in range(min1_idx-1):
                diff = np.random.normal(pre_diff_mean, pre_diff_std)
                pre_seq.append(pre_seq[-1]+diff)

            pre_seq = pre_seq[::-1]
            synthetic_sequences[:len(pre_seq), i] = pre_seq

            synthetic_sequences[len(pre_seq):len(pre_seq)+len(lorentzian_seq), i] = lorentzian_seq

            post_diff_mean = np.random.normal(post_mean_mean_diffs, post_std_mean_diffs)
            post_diff_std = np.abs(np.random.normal(post_mean_std_diffs, post_std_std_diffs))
            post_seq = [lorentzian_seq[-1]]
            for j in range(68 - min2_idx - 1):
                diff = np.random.normal(post_diff_mean, post_diff_std)
                post_seq.append(post_seq[-1]+diff)

            synthetic_sequences[min2_idx:, i] = post_seq[:68-min2_idx]

            seq = synthetic_sequences[:, i]
            noise_std_high = noise_std*gen_A*2
            noise_std_med = noise_std*gen_A
            noise_std_low = noise_std*gen_A*0.5
            random_noise_std = np.random.choice([noise_std_high, noise_std_med, noise_std_low])

            synthetic_sequences_noisy[:, i] = seq + np.random.normal(0, random_noise_std, size=seq.shape[0])

        self.synthetic_data = {
            "synthetic_sequences": synthetic_sequences_noisy,
            "bfs_xs": bfs_xs,
            "fwhms": fwhms
        }
    
    def model_architecture(self):
        print("Creating dual-output CNN model...")
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

        # FWHM prediction branch (regression)
        fwhm_branch = Dense(128, activation='relu')(x)
        fwhm_branch = Dense(64, activation='relu')(fwhm_branch)
        fwhm_branch = BatchNormalization()(fwhm_branch)
        fwhm_output = Dense(1, name='fwhm_output')(fwhm_branch)

        model = Model(inputs=inputs, outputs=[bfs_output, fwhm_output])

        print("Model architecture created successfully!")
        model.summary()
        self.model = model

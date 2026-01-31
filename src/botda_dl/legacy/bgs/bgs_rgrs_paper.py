from .bgs import BGS
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import preprocessing as prepro


def bgs_data_gen(wvec, bgs_parms, noise=0.0):
    n_w = len(wvec)
    bfs, fwhm = bgs_parms
    amp = 1.0
    func = amp / (1 + ((wvec - bfs) / (fwhm / 2)) ** 2)
    func = func + noise * np.random.normal(0, 1, n_w)
    return func

def bgs_scale(yvec, scaling='Standard'):
    if scaling == 'Standard':
        scaler = prepro.StandardScaler()
    elif scaling == 'MinMax':
        scaler = prepro.MinMaxScaler()
    elif scaling == 'Robust':
        scaler = prepro.RobustScaler()
    else:
        raise ValueError()
    return scaler.fit_transform(yvec.reshape(len(yvec), 1)).flatten()

def LS_fit(data, x0):
    xdata = data[:, 0]
    ydataraw = data[:, 1]
    ydata = bgs_scale(ydataraw, 'MinMax')
    def fun(xdata, fpeak, fwhm):
        return bgs_data_gen(xdata, [fpeak, fwhm])
    x, cov_x = curve_fit(fun, xdata, ydata, p0=x0, bounds=(np.array([0, 5]), np.array([80, 60])))
    x_err = [np.sqrt(cov_x[j, j]) for j in range(x.size)]
    return x, x_err

def batch_LS_fit(raw_data, wvec, x0):
    n_x = raw_data.shape[1]
    parmat = np.zeros((n_x, 2))
    parerr = np.zeros((n_x, 2))
    for i in tqdm(range(n_x)):
        data = np.column_stack([wvec, raw_data[:, i]])
        parmat[i, :], parerr[i, :] = LS_fit(data, x0)
    return parmat, parerr

class BGS_PAPER_BASE:
    def __init__(self, wvec, scaling='Standard', batch_size=100, max_epochs=500):
        self.wvec = wvec
        self.scaling = scaling
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.input_shape = (len(wvec), 2)
        self.n_out = 2
        self.model = None
        self.model_history = None

    def data_batchgen(self, parmat):
        n_w, n_c = self.input_shape
        N = parmat.shape[0]
        func_parms = parmat[:, 0:self.n_out]
        func_results = np.zeros(shape=(N, n_w, n_c))
        for i in range(N):
            func_results[i, :, 0] = self.wvec
            bgsvec = bgs_data_gen(self.wvec, parmat[i, :2], noise=1 / parmat[i, 2])
            func_results[i, :, 1] = bgs_scale(bgsvec, scaling=self.scaling)
        return func_results, func_parms

    def create_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        x = layers.Conv1D(64, 15, activation='relu', padding='same')(x)
        x = layers.MaxPool1D(3)(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.MaxPool1D(3)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu', bias_regularizer='l2', kernel_regularizer='l2')(x)
        x = layers.Dense(2, activation='linear')(x)  # Output: [bfs, fwhm]
        self.model = tf.keras.Model(inputs, x)

    def train_model(self, N=50000, X=None, y=None, par_min=(10.0, 5.0, 5.0), par_max=(55.0, 40.0, 18.0)):
        if X is None:
            parmat = np.random.uniform(par_min, par_max, (N, 3))
            X, y = self.data_batchgen(parmat)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mse')
        self.model_history = self.model.fit(X, y, validation_split=0.3, batch_size=self.batch_size, epochs=self.max_epochs, shuffle=True, verbose=2)

    def batch_predict(self, data):
        n_w, n_c = self.input_shape
        N = data.shape[1]
        inputs = np.zeros((N, n_w, n_c))
        for i in range(N):
            inputs[i, :, 0] = self.wvec
            inputs[i, :, 1] = bgs_scale(data[:, i], scaling=self.scaling)
        preds = self.model.predict(inputs)
        return preds, np.zeros_like(preds)  # No stddev, just zeros


class BGS_RGRS_PAPER(BGS):
    def lorentzian(self, x, A, bfs_x, gamma, offset):
        return offset + (A * gamma**2) / ((x - bfs_x)**2 + gamma**2)
    
    def fit_lorentzian_freq_gain(self, x, y):
        try:
            A_guess = np.max(y)

            bfs_x_guess_idx = np.round(x.shape[0]/2).astype(int)
            bfs_x_guess = self.idx_to_freq(bfs_x_guess_idx)

            gamma_guess_idx = np.round(x.shape[0]/6).astype(int)
            gamma_guess = self.idx_to_freq(gamma_guess_idx) - self.idx_to_freq(0)

            offset_guess = np.min(y)

            params, _ = curve_fit(self.lorentzian, x, y,
                                p0=[A_guess, bfs_x_guess , gamma_guess, offset_guess],
                                maxfev=5000)
        except RuntimeError:
            print('Lorentzian fit failed - using initial guesses')
            params = [A_guess, bfs_x_guess , gamma_guess, offset_guess]

        return {"A": float(params[0]),
                "bfs_x": float(params[1]),
                "gamma": float(params[2]),
                "offset": float(params[3])}

    def single_sequence_analyze(self, sequence, start_idx=0):
        sequences_len = sequence.shape[0]
        idx_x = np.arange(start_idx, sequences_len+start_idx)
        freq_x = self.idx_to_freq(idx_x)

        peak_argmax_idx = np.array([np.argmax(sequence)])
        peak_argmax_x = self.idx_to_freq(peak_argmax_idx)[0]

        lor_params = self.fit_lorentzian_freq_gain(freq_x, sequence)

        fitted_lorentzian_params = lor_params['A'], lor_params['bfs_x'], lor_params['gamma'], lor_params['offset']

        fitted_lor_freq_gain = self.lorentzian(freq_x, *fitted_lorentzian_params)

        bfs_lor_x = lor_params['bfs_x']
        lor_fwhm = 2 * lor_params['gamma']

        fwhm_left_x = bfs_lor_x - lor_params['gamma']
        fwhm_right_x = bfs_lor_x + lor_params['gamma']

        return {
            "bfs_lor_x": bfs_lor_x,
            "peak_argmax_x": peak_argmax_x,
            "lor_fwhm": lor_fwhm,
            "fwhm_left_x": fwhm_left_x,
            "fwhm_right_x": fwhm_right_x,
            "fitted_lor_freq_gain": fitted_lor_freq_gain,
            "fitted_lorentzian_params": fitted_lorentzian_params
        }
    
    def analyze(self, title="Real Data", plot=True):
        n_sequences = self.data.shape[1]

        argmax_peak_frequencies = np.zeros((n_sequences))
        lorentz_bfs_frequencies = np.zeros((n_sequences))
        fwhms = np.zeros((n_sequences))

        lorentzian_params = np.zeros((n_sequences, 4))

        for i in tqdm(range(n_sequences), desc=f"analyzing {title} sequences"):
            sequence = self.data[:, i].flatten()

            res = self.single_sequence_analyze(sequence)

            bfs_lor_x = res["bfs_lor_x"]
            peak_argmax_x = res["peak_argmax_x"]
            lor_fwhm = res["lor_fwhm"]
            fitted_lorentzian_params = res["fitted_lorentzian_params"]

            argmax_peak_frequencies[i] = peak_argmax_x
            lorentz_bfs_frequencies[i] = bfs_lor_x
            fwhms[i] = lor_fwhm
            lorentzian_params[i, :] = fitted_lorentzian_params

        lorentzian_stats = {
            'mean_A': np.mean(lorentzian_params[:, 0]),
            'std_A': np.std(lorentzian_params[:, 0]),

            'mean_bfs_x': np.mean(lorentzian_params[:, 1]),
            'std_bfs_x': np.std(lorentzian_params[:, 1]),

            'mean_gamma': np.mean(lorentzian_params[:, 2]),
            'std_gamma': np.std(lorentzian_params[:, 2]),

            'mean_offset': np.mean(lorentzian_params[:, 3]),
            'std_offset': np.std(lorentzian_params[:, 3]),

        }

        if plot:
            plt.figure(figsize=(14, 12))
            palette = sns.color_palette('colorblind', 6)

            plt.subplot(2, 3, 1)
            sns.histplot(argmax_peak_frequencies, bins=20, kde=True, color=palette[0], label='Distribution', linewidth=2)
            plt.title(f'{title} argmax Peak Frequency Distribution', fontsize=18)
            plt.xlabel('Frequency (MHz)', fontsize=16)
            plt.ylabel('Count', fontsize=16)
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.savefig(f'{self.results_dir}/{title}_argmax_peak_freq_dist.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_argmax_peak_freq_dist.png', format='png', dpi=300)

            plt.subplot(2, 3, 2)
            sns.histplot(lorentz_bfs_frequencies, bins=20, kde=True, color=palette[1], label='Distribution', linewidth=2)
            plt.title(f'{title} Lorentzian BFS Distribution', fontsize=18)
            plt.xlabel('BFS (MHz)', fontsize=16)
            plt.ylabel('Count', fontsize=16)
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_bfs_dist.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_bfs_dist.png', format='png', dpi=300)

            plt.subplot(2, 3, 3)
            sns.histplot(fwhms, bins=20, kde=True, color=palette[2], label='Distribution', linewidth=2)
            plt.title(f'{title} FWHM Distribution', fontsize=18)
            plt.xlabel('FWHM', fontsize=16)
            plt.ylabel('Count', fontsize=16)
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.savefig(f'{self.results_dir}/{title}_fwhm_dist.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_fwhm_dist.png', format='png', dpi=300)

            plt.subplot(2, 3, 4)
            sns.scatterplot(x=lorentz_bfs_frequencies, y=fwhms, alpha=0.7, color=palette[3], label='Data points', s=40, edgecolor='k')
            plt.title(f'{title} Lorentzian BFS vs FWHM', fontsize=18)
            plt.xlabel('BFS (MHz)', fontsize=16)
            plt.ylabel('FWHM', fontsize=16)
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_vs_fwhm.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_vs_fwhm.png', format='png', dpi=300)

            plt.subplot(2, 3, 5)
            sns.scatterplot(x=lorentz_bfs_frequencies, y=argmax_peak_frequencies, alpha=0.7, color=palette[4], label='Data points', s=40, edgecolor='k')
            plt.title(f'{title} Lorentzian vs argmax Peak Frequency', fontsize=18)
            plt.xlabel('Lorentzian BFS (MHz)', fontsize=16)
            plt.ylabel('argmax Frequency (MHz)', fontsize=16)
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_vs_argmax.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_lorentz_vs_argmax.png', format='png', dpi=300)

            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/{title}_all_plots.svg', format='svg')
            plt.savefig(f'{self.results_dir}/{title}_all_plots.png', format='png', dpi=300)
            plt.show()

        self.analyze_results =  {
            "lorentzian_stats": lorentzian_stats,
            "argmax_peak_frequencies": argmax_peak_frequencies,
            "lorentz_bfs_frequencies": lorentz_bfs_frequencies,
            "fwhms": fwhms
        }
    
    def gen_synth(self, n=10000, noise_std=0.01):

        lorentzian_stats = self.analyze_results["lorentzian_stats"]

        mean_A = lorentzian_stats['mean_A']
        std_A = lorentzian_stats['std_A']

        mean_bfs_x = lorentzian_stats['mean_bfs_x']
        std_bfs_x = lorentzian_stats['std_bfs_x']

        mean_gamma = lorentzian_stats['mean_gamma']
        std_gamma = lorentzian_stats['std_gamma']

        mean_offset = lorentzian_stats['mean_offset']
        std_offset = lorentzian_stats['std_offset']


        synthetic_sequences = np.zeros((68, n))
        synthetic_bfs_frequencies = np.zeros((n))
        synthetic_fwhms = np.zeros((n))

        for i in range(n):
            synth_A = np.random.normal(mean_A, std_A)
            synth_bfs_x = np.random.normal(mean_bfs_x, std_bfs_x)
            synth_gamma = np.random.normal(mean_gamma, std_gamma)
            synth_offset = np.random.normal(mean_offset, std_offset)

            synth_lor_y = self.lorentzian(self.frequency_axis_mhz, synth_A, synth_bfs_x, synth_gamma, synth_offset)

            synth_noise_std = noise_std * np.abs(synth_A)

            noisy_lor_y = np.maximum(0, synth_lor_y+np.random.normal(0, synth_noise_std, size=synth_lor_y.shape)).flatten()

            synthetic_sequences[:, i] = noisy_lor_y
            synthetic_bfs_frequencies[i] = synth_bfs_x
            synthetic_fwhms[i] = 2 * synth_gamma


        self.synthetic_data = {
            "synthetic_sequences": synthetic_sequences,
            "bfs_xs": synthetic_bfs_frequencies,
            "fwhms": synthetic_fwhms
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

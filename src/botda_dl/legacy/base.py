from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from .config import *

class BaseObject(ABC):
    def __init__(self, data_path, model_path, log_dir, results_dir, scalers_dir):

        self.data_path = data_path
        self.model_path = model_path
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.scalers_dir = scalers_dir

        self.data = None
        self.model = None
        self.callbacks = None


    def read_file(self):
        print(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r') as f:
            lines = f.readlines()

        data = np.array([list(map(float, line.strip().split(','))) for line in lines])
        data = data[:, ~np.isnan(data).any(axis=0)]
        self.data = data
        print(f"Loaded data shape: {data.shape}")

        actual_num_freq_points, actual_num_distance_points = self.data.shape
        print(f"Detected data shape: {actual_num_freq_points} frequency points (rows)\
              and {actual_num_distance_points} distance points (columns).")
        self.distance_axis_idx = np.arange(actual_num_distance_points)
        self.frequency_axis_idx = np.arange(actual_num_freq_points)
        self.frequency_axis_mhz = np.linspace(FREQUENCY_START_MHZ, FREQUENCY_END_MHZ, actual_num_freq_points)

    def idx_to_freq(self, idx):
        return self.frequency_axis_mhz[idx]

    def freq_to_idx(self, freq):
        idx_array = None
        try:
            idx_array = np.zeros((freq.shape[0]))
            for i in range(freq.shape[0]):
                closest_idx = np.argmin(np.abs(self.frequency_axis_mhz - freq[i]))
                idx_array[i] = closest_idx
        except:
            print("its not an array of frequencies! make it array!")

        return np.round(idx_array).astype(int)

    def create_model(self):
        self.model_architecture()
        self.compile_model()
        print(f"Model created: {self.model}")

    def train_and_visualize(self, train_X, train_Y, test_X, test_Y,
                        epochs=512, batch_size=1024):

        print(f"Starting training for {epochs} epochs...")

        history = self.model.fit(
            x=train_X,
            y=train_Y,
            validation_data=(test_X, test_Y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
            verbose=1
        )

        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")
        palette = sns.color_palette("colorblind", 8)

        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Train', color=palette[0], linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', color=palette[1], linewidth=2)
        plt.title('Total Loss (Weighted Sum)', fontsize=18)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(frameon=False, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 2, 2)
        # Check if this is a dual-output model or single-output model
        if 'bfs_loss' in history.history:
            plt.plot(history.history['bfs_loss'], label='Train', color=palette[2], linewidth=2)
            plt.plot(history.history['val_bfs_loss'], label='Validation', color=palette[3], linewidth=2)
            plt.title('BFS Output Loss', fontsize=14)
        elif 'loss' in history.history and len(history.history.keys()) == 2:
            # Single output model - just show total loss
            plt.plot(history.history['loss'], label='Train', color=palette[2], linewidth=2)
            plt.plot(history.history['val_loss'], label='Validation', color=palette[3], linewidth=2)
            plt.title('Total Loss', fontsize=14)
        else:
            # Fallback for any other case
            plt.plot(history.history['loss'], label='Train', color=palette[2], linewidth=2)
            plt.plot(history.history['val_loss'], label='Validation', color=palette[3], linewidth=2)
            plt.title('Total Loss', fontsize=14)
        plt.ylabel('Loss', fontsize=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.legend(frameon=False, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 2, 3)
        if 'fwhm_loss' in history.history:
            plt.plot(history.history['fwhm_loss'], label='Train', color=palette[4], linewidth=2)
            plt.plot(history.history['val_fwhm_loss'], label='Validation', color=palette[5], linewidth=2)
            plt.title('FWHM Output Loss', fontsize=14)
        else:
            # Single output model - show empty plot or skip
            plt.text(0.5, 0.5, 'Single Output Model\n(No FWHM)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('FWHM Output Loss (N/A)', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(frameon=False, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 2, 4)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate', color=palette[6], linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14)
            plt.ylabel('LR', fontsize=12)
            plt.xlabel('Epoch', fontsize=12)
            plt.yscale('log')
            plt.legend(frameon=False, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/training_history.svg', format='svg')
        plt.savefig(f'{self.results_dir}/training_history.png', format='png', dpi=300)
        plt.show()

        print("Training complete. Final val loss:", history.history['val_loss'][-1])
        print("\nSaving fitting history ...")
        with open(f'{self.results_dir}/history.pkl', 'wb') as f:
            pickle.dump(history, f)
    
    def full_pipeline(self, n_synthetic=10000, epochs=32, batch_size=512):
        print("=== Running Full Pipeline ===")
        self.read_file()

        print("Analyzing real data statistics...")
        self.analyze()

        print(f"\nGenerating {n_synthetic} synthetic sequences...")
        self.gen_synth(n=n_synthetic)

        print("\nPreprocessing data for model training...")
        X_train, X_test, y_train, y_test = self.preprocess_data()

        self.create_model()

        print(f"\nTraining model for {epochs} epochs...")
        self.train_and_visualize(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size
        )


    @abstractmethod
    def model_architecture(self):
        pass

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def gen_synth(self):
        pass

"""
Base class for BOTDA spectral analysis using deep learning.
Provides common functionality for BGS and BPS analysis.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

from ..utils.config import FREQUENCY_START_MHZ, FREQUENCY_END_MHZ, FREQUENCY_RESOLUTION


class BaseAnalyzer(ABC):
    """
    Abstract base class for BOTDA spectrum analysis.
    
    Handles data loading, model training, visualization, and synthetic data generation.
    """

    def __init__(self, data_path, model_path, log_dir, results_dir, scalers_dir):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to input data file
            model_path: Path to save/load trained model
            log_dir: Directory for training logs
            results_dir: Directory for results and plots
            scalers_dir: Directory for saving data scalers
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.scalers_dir = Path(scalers_dir)

        self.data = None
        self.model = None
        self.callbacks = None
        self.synthetic_data = None
        self.analyze_results = None

        # Initialize frequency axis
        self.frequency_axis_mhz = np.linspace(
            FREQUENCY_START_MHZ, 
            FREQUENCY_END_MHZ, 
            FREQUENCY_RESOLUTION
        )
        self.distance_axis_idx = None
        self.frequency_axis_idx = None

        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create output directories if they do not exist."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scalers_dir.mkdir(parents=True, exist_ok=True)

    def read_file(self):
        """Load data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Input data file not found: {self.data_path}")

        print(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            first_non_empty = ""
            for line in f:
                candidate = line.strip()
                if candidate:
                    first_non_empty = candidate
                    break

        if not first_non_empty:
            raise ValueError(f"Input data file is empty: {self.data_path}")

        delimiter = "," if "," in first_non_empty else None
        data = np.genfromtxt(self.data_path, delimiter=delimiter)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if np.isnan(data).all():
            raise ValueError(f"Input data file only contains NaNs: {self.data_path}")

        if np.isnan(data).any():
            data = data[:, ~np.isnan(data).any(axis=0)]
            if data.size == 0:
                raise ValueError(f"All columns contain NaNs in {self.data_path}")

        self.data = data

        print(f"Loaded data shape: {data.shape}")
        actual_num_freq_points, actual_num_distance_points = self.data.shape
        print(f"Detected data shape: {actual_num_freq_points} frequency points (rows) "
              f"and {actual_num_distance_points} distance points (columns).")

        self.distance_axis_idx = np.arange(actual_num_distance_points)
        self.frequency_axis_idx = np.arange(actual_num_freq_points)

        if actual_num_freq_points != self.frequency_axis_mhz.size:
            print(
                "Warning: frequency resolution mismatch. "
                f"Expected {self.frequency_axis_mhz.size}, got {actual_num_freq_points}. "
                "Rebuilding frequency axis."
            )
            self.frequency_axis_mhz = np.linspace(
                FREQUENCY_START_MHZ,
                FREQUENCY_END_MHZ,
                actual_num_freq_points
            )

    def idx_to_freq(self, idx):
        """Convert frequency index to MHz value."""
        return self.frequency_axis_mhz[idx]

    def freq_to_idx(self, freq):
        """Convert frequency values to indices."""
        if self.frequency_axis_mhz is None:
            raise ValueError("Frequency axis is not initialized.")

        if isinstance(freq, np.ndarray) and freq.ndim > 0:
            idx_array = np.zeros((freq.shape[0]))
            for i in range(freq.shape[0]):
                closest_idx = np.argmin(np.abs(self.frequency_axis_mhz - freq[i]))
                idx_array[i] = closest_idx
        else:
            closest_idx = np.argmin(np.abs(self.frequency_axis_mhz - freq))
            idx_array = np.array([closest_idx])

        return np.round(idx_array).astype(int)

    def _build_callbacks(self, monitor="val_loss", patience=32, reduce_patience=8):
        """Create default training callbacks with LR logging."""
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                verbose=1,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_patience,
                verbose=1,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                str(self.model_path),
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                update_freq='epoch',
                write_graph=True,
                write_images=True
            )
        ]

        class LRTracker(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self._lrs = []

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current_lr = keras.backend.get_value(self.model.optimizer.learning_rate)
                self._lrs.append(current_lr)
                logs['lr'] = current_lr

        callbacks.append(LRTracker())
        return callbacks

    def create_model(self):
        """Create and compile the model."""
        self.model_architecture()
        self.compile_model()
        
        plot_file = self.results_dir / 'model_plot.png'
        try:
            from tensorflow.keras.utils import plot_model as tf_plot_model
            tf_plot_model(self.model, to_file=str(plot_file), 
                         show_shapes=True, show_layer_names=True)
        except Exception as e:
            print(f"Could not save model plot: {e}")
        
        print(f"Model created successfully")
        self.model.summary()

    def train_and_visualize(self, train_X, train_Y, test_X, test_Y,
                           epochs=512, batch_size=1024):
        """
        Train the model and visualize training history.
        
        Args:
            train_X: Training input data
            train_Y: Training target data
            test_X: Test input data
            test_Y: Test target data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
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

        # Visualize training history
        self._plot_training_history(history)

        print("Training complete. Final val loss:", history.history['val_loss'][-1])
        print("\nSaving training history...")
        
        history_file = self.results_dir / 'history.pkl'
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)

    def _plot_training_history(self, history):
        """Plot training and validation metrics."""
        plt.figure(figsize=(18, 12))
        sns.set_style("whitegrid")
        palette = sns.color_palette("husl", 8)

        # Total loss
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Train', color=palette[0])
        plt.plot(history.history['val_loss'], label='Validation', color=palette[1])
        plt.title('Total Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Loss by output (if available)
        if 'peak_output_loss' in history.history:
            plt.subplot(2, 2, 2)
            plt.plot(history.history['peak_output_loss'], label='Train', color=palette[2])
            plt.plot(history.history['val_peak_output_loss'], label='Validation', color=palette[3])
            plt.title('Peak Output Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            plt.subplot(2, 2, 3)
            if 'fwhm_output_loss' in history.history:
                plt.plot(history.history['fwhm_output_loss'], label='Train', color=palette[4])
                plt.plot(history.history['val_fwhm_output_loss'], label='Validation', color=palette[5])
                plt.title('FWHM Output Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend()

        # Learning rate (if available)
        plt.subplot(2, 2, 4)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate', color=palette[6])
            plt.title('Learning Rate Schedule')
            plt.ylabel('LR')
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=100)
        plt.show()

    def full_pipeline(self, n_synthetic=300000, epochs=128, batch_size=2048):
        """
        Execute the complete analysis pipeline.
        
        Args:
            n_synthetic: Number of synthetic samples to generate
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("\n" + "="*60)
        print("Running Full BOTDA Analysis Pipeline")
        print("="*60)
        
        self.read_file()

        print("\nAnalyzing real data statistics...")
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
        
        print("\n" + "="*60)
        print("Pipeline execution completed!")
        print("="*60)

    @abstractmethod
    def model_architecture(self):
        """Define the neural network architecture."""
        pass

    @abstractmethod
    def compile_model(self):
        """Compile the model with optimizer and loss function."""
        pass

    @abstractmethod
    def analyze(self):
        """Analyze real data and extract statistics."""
        pass

    @abstractmethod
    def gen_synth(self, n=1000):
        """Generate synthetic data based on real data statistics."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Preprocess and normalize data for training."""
        pass

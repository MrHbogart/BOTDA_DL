from ..base import BaseObject
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                            ModelCheckpoint, TensorBoard)
import seaborn as sns


class BPS(BaseObject):
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        print("Starting data preprocessing...")

        sequences = self.synthetic_data["synthetic_sequences"]
        bfs_xs = self.synthetic_data["bfs_xs"]
        

        normalized_sequences = np.zeros_like(sequences)

        for i in tqdm(range(sequences.shape[1]), desc="Normalizing sequences"):
            sequence = sequences[:, i]
            scaler = MinMaxScaler()
            normalized_sequences[:, i] = scaler.fit_transform(sequence.reshape(-1, 1)).flatten()

        X = normalized_sequences.T.reshape((-1, 68, 1))

        bfs_scaler = StandardScaler()
        normalized_bfs = bfs_scaler.fit_transform(bfs_xs.reshape(-1, 1)).flatten()
        print(np.mean(normalized_bfs))
        print(np.std(normalized_bfs))

        X_train, X_test, y_train, y_test = train_test_split(
            X, normalized_bfs, test_size=test_size, random_state=random_state
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Visualizations with more plots
        palette = sns.color_palette('colorblind', 6)
        plt.figure(figsize=(16, 10))

        sample_idx = np.random.randint(0, sequences.shape[1])
        plt.subplot(2, 3, 1)
        plt.plot(self.frequency_axis_mhz, normalized_sequences[:, sample_idx], label='Normalized', color=palette[0], linewidth=2)
        plt.axvline(x=bfs_xs[sample_idx], color=palette[1], linestyle='--', label='Peak', linewidth=2)
        plt.title(f'Sample Sequence (Index {sample_idx})', fontsize=12)
        plt.xlabel('Position', fontsize=10)
        plt.ylabel('Value', fontsize=10)
        plt.legend(frameon=False, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 3, 2)
        plt.hist(bfs_xs, bins=68, color=palette[2], edgecolor='black', alpha=0.8)
        plt.title('Distribution of BFSs', fontsize=12)
        plt.xlabel('BFS', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 3, 4)
        for i in range(min(5, sequences.shape[1])):
            plt.plot(self.frequency_axis_mhz, normalized_sequences[:, i], alpha=0.5, color=palette[3], linewidth=1.5)
            plt.axvline(x=bfs_xs[i], color=palette[4], linestyle='--', alpha=0.3, linewidth=1.5)
        plt.title('Multiple Normalized Sequences with Peaks', fontsize=12)
        plt.xlabel('Position', fontsize=10)
        plt.ylabel('Normalized Value', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.subplot(2, 3, 5)
        plt.bar(['Train', 'Test'], [len(X_train), len(X_test)], color=palette[5], edgecolor='black', alpha=0.8)
        plt.title('Train-Test Split', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/preprocessing_overview.svg', format='svg')
        plt.savefig(f'{self.results_dir}/preprocessing_overview.png', format='png', dpi=300)
        plt.show()

        print("\nSaving scaler for future use...")
        with open(f'{self.scalers_dir}/bfs_scaler.pkl', 'wb') as f:
            pickle.dump(bfs_scaler, f)

        print("Preprocessing completed.")

        return X_train, X_test, y_train, y_test
    
    def compile_model(self, initial_lr=0.001):

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss={
                'bfs_output': 'mse',
            },
            metrics={
                'bfs_output': 'mae',
            }
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=32,
                verbose=1,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                verbose=1,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=self.log_dir,
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

        lr_tracker = None
        for cb in callbacks:
            if isinstance(cb, LRTracker):
                lr_tracker = cb
                break

        if lr_tracker is None:
            lr_tracker = LRTracker()
            callbacks.append(lr_tracker)

        self.callbacks = callbacks
        print("Model compiled with callbacks:", [cb.__class__.__name__ for cb in callbacks])

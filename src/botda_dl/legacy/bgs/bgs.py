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


class BGS(BaseObject):
    def preprocess_data(self, test_size=0.2, random_state=42):
        sequences = self.synthetic_data["synthetic_sequences"]
        bfs = self.synthetic_data["bfs_xs"]
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

        # Scale bfs values
        bfs_scaler = StandardScaler()
        normalized_bfs = bfs_scaler.fit_transform(bfs.reshape(-1, 1)).flatten()

        # Scale FWHM values
        fwhms_scaler = StandardScaler()
        normalized_fwhms = fwhms_scaler.fit_transform(fwhms.reshape(-1, 1)).flatten()

        # Train-test split
        X_train, X_test, y_bfs_train, y_bfs_test, y_fwhm_train, y_fwhm_test = train_test_split(
            X, normalized_bfs, normalized_fwhms, test_size=test_size, random_state=random_state
        )

        y_train = {
            'bfs_output': y_bfs_train,
            'fwhm_output': y_fwhm_train
        }
        y_test = {
            'bfs_output': y_bfs_test,
            'fwhm_output': y_fwhm_test
        }


        plt.figure(figsize=(18, 12))
        plot_cols = 2
        plot_rows = 3
        n_samples = plot_cols*plot_rows

        for i in range(n_samples):
            sample_idx = np.random.randint(0, X.shape[0])

            normalized_sequence = X[sample_idx].flatten()
            normal_bfs = normalized_bfs[sample_idx]
            normalized_fwhm = normalized_fwhms[sample_idx]

            bfs_val = bfs_scaler.inverse_transform(normal_bfs.reshape((-1, 1))).flatten()[0]
            fwhm = fwhms_scaler.inverse_transform(normalized_fwhm.reshape((-1, 1))).flatten()[0]


            plt.subplot(plot_rows, plot_cols, i+1)

            plt.plot(self.frequency_axis_mhz, normalized_sequence, label='Normalized')
            plt.axvline(bfs_val, color='r', linestyle='--', label='Peak')
            plt.title(f'Sample Sequence (Peak: {bfs_val:.2f}, FWHM: {fwhm:.2f})', fontsize=12)
            plt.legend()

        plt.tight_layout()
        plt.show()

        print("Preprocessing complete. Train shape:", X_train.shape)

        print("\nSaving scalers for future use...")
        with open(f'{self.scalers_dir}/fwhms_scaler.pkl', 'wb') as f:
            pickle.dump(fwhms_scaler, f)
        with open(f'{self.scalers_dir}/bfs_scaler.pkl', 'wb') as f:
            pickle.dump(bfs_scaler, f)


        return X_train, X_test, y_train, y_test

    def compile_model(self, initial_lr=0.001):
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=initial_lr),
            loss={
                'bfs_output': 'mse',
                'fwhm_output': 'mse'
            },
            loss_weights={
                'bfs_output': 0.5,
                'fwhm_output': 0.5
            },
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

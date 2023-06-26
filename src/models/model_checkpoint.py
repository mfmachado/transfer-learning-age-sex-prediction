from tensorflow.python.keras.callbacks import ModelCheckpoint


class ModelCheckpointWarmUp(ModelCheckpoint):

    def __init__(self, filepath, warmup_epochs: int, **kwargs):
        super().__init__(filepath, **kwargs)
        self.warmup_epochs = warmup_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.warmup_epochs <= epoch:
            if self.save_freq == 'epoch':
                self._save_model(epoch=epoch, logs=logs)

class Callback:
    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_batch_end(self, trainer, loss):
        pass

    def on_evaluation_end(self, trainer, evaluation_results):
        pass

    def on_teardown(self, trainer):
        pass

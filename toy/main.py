import shutil
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import RichProgressBar, Callback
from lightning.pytorch.cli import LightningCLI


class GlobalStepRichProgressBar(RichProgressBar):
    @property
    def total_train_batches(self):
        return self.trainer.max_steps


class MyLightningCLI(LightningCLI):
    def after_fit(self):
        self.trainer.test(self.model, self.datamodule)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    cli = MyLightningCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        trainer_defaults = {
            "callbacks": [lazy_instance(GlobalStepRichProgressBar)],
        }
    )

import shutil
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import RichProgressBar, Callback
from lightning.pytorch.cli import LightningCLI


class GlobalStepRichProgressBar(RichProgressBar):
    @property
    def total_train_batches(self):
        return self.trainer.max_steps - self.trainer.global_step


class ClearLogdir(Callback):
    def on_fit_start(self, trainer, pl_module):
        log_dir = Path(trainer.logger.log_dir)
        if log_dir.exists():
            targets = log_dir.glob("*")
            for target in targets:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.is_file():
                    target.unlink()
        Path(trainer.log_dir).mkdir(exist_ok=True, parents=True)


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
        # parser_kwargs={"parser_mode": "omegaconf"},
    )

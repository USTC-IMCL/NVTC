# NVTC for Lossy Image Coding

This directory provides the training/testing code of our NVTC models.


Todo List:
* [x] Basic training framework.
* [x] Testing code and pretrained models.
* [ ] Practical entropy coding.


## Installation
We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) for installation.

```bash
python -m pip install -r requirements.txt
pip install -e .
```

## Usage
Our training framework and command line interface are based on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

### Testing
The pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1uEj3dQImMiH0RjikUZ-h9qoi4NAyL2j3?usp=sharing) at google drive.
To test the pretrained model with lambda 128, run:

```bash
python main.py test \
--config ckpt/m128/config.yaml \
--ckpt_path ckpt/m128/last_pure.ckpt \
--data.test_glob "/your/image/path/in/glob/pattern"
```
Specifically, the `config` file includes all the configurations to train and initialize the model.
And the `test_glob` is the path of test images in glob pattern, e.g., `"kodak/*.png"`.
The test results will be saved at `./log` by default. 
You can freely change the saving directory by adding `--trainer.logger.save_dir /your/save/dir`.

### Training
The example command lines for training can be found in [script/](script) directory.
To train the model with lambda 128, run:
```bash
python main.py fit \
--data config/data/coco2017.yaml \
--model config/model/nvtc.yaml \
--model.lmbda 128 \
--trainer config/trainer.yaml \
--trainer.logger.save_dir ./log \
--trainer.logger.name nvtc \
--trainer.logger.version m128
```
Note that you should customize your own training and validation dataset 
by rewriting [config/data/coco2017.yaml](config/data/coco2017.yaml) and [nvtc_image/data.py](nvtc_image/data.py).
In this repo, we use python-lmdb to build our training dataset, 
where the example script to make the lmdb dataset is provided in [nvtc_image/write_data.py](nvtc_image%2Fwrite_data.py).

You can freely change the configuration file for your research propose following the rules of pytorch lightning.

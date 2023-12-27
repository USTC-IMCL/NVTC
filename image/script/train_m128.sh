export JSONARGPARSE_DEBUG=0

export CUDA_VISIBLE_DEVICES="0"
python main.py fit \
--data config/data/coco2017.yaml \
--model config/model/nvtc.yaml \
--model.lmbda 128 \
--trainer config/trainer.yaml \
--trainer.logger.save_dir ./log \
--trainer.logger.name nvtc \
--trainer.logger.version m128
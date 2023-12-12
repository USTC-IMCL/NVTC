export CUDA_VISIBLE_DEVICES="0"
group=vector2d
source=boomerang
model=ntc
lmbda=64

python main.py fit \
--data config/data/${group}/${source}.yaml \
--model config/model/${group}/${model}.yaml \
--model.lmbda ${lmbda} \
--trainer config/trainer.yaml \
--trainer.val_check_interval 10000 \
--trainer.logger.save_dir ./log \
--trainer.logger.name ${group} \
--trainer.logger.version ${source}-${model}-m${lmbda}

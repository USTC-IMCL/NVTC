cd ../..

# 38
save_dir=/data/fengrs/compression2023/LSVQ/logs

export CUDA_VISIBLE_DEVICES="5"
group=vector4d
source=normal
model=ntc
for lmbda in 128
do
  python main.py fit \
  --data config/data/${group}/${source}.yaml \
  --model config/model/${group}/${model}.yaml \
  --model.lmbda ${lmbda} \
  --trainer config/trainer.yaml \
  --trainer.val_check_interval 10000 \
  --trainer.logger.save_dir ${save_dir} \
  --trainer.logger.name ${group} \
  --trainer.logger.version ${source}-${model}-m${lmbda} \
#  > log/${group}-${source}-${model}-m${lmbda}.log
done

cd ../..

# 38
save_dir=/data/fengrs/compression2023/LSVQ/logs

export CUDA_VISIBLE_DEVICES="5"
group=vector4d
source=normal
model=ecvq
ncb=32768
for lmbda in 1024
do
  python main.py fit \
  --data config/data/${group}/${source}.yaml \
  --model config/model/${group}/${model}.yaml \
  --model.lmbda ${lmbda} \
  --model.cb_size ${ncb} \
  --trainer config/trainer.yaml \
  --trainer.val_check_interval 10000 \
  --trainer.logger.save_dir ${save_dir} \
  --trainer.logger.name ${group} \
  --trainer.logger.version ${source}-${model}-m${lmbda}-ncb${ncb} \
#  > log/${group}-${source}-${model}-m${lmbda}-ncb${ncb}.log
done
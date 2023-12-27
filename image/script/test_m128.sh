export JSONARGPARSE_DEBUG=0

export CUDA_VISIBLE_DEVICES="0"
python main.py test --config /your_save_dir/config.yaml --ckpt_path /your_save_dir/checkpoints/last.ckpt
python -u main.py \
  --model incoder \
  --model_args pretrained=/checkpoint/dpf/models/recoder/1.3B-CM-init-hf/epoch-2 \
  --device 0 \
  --batch_size 1 \
  --tasks recoder \
  --limit 5000 \
  --no_cache \
  | tee expts/epoch-2_recoder-5K.out

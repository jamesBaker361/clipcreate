  python3 ddpo_train_script.py --pretrained_model_name_or_path "jlbaker361/sd-wikiart-lora-balanced100" \
  --image_dir "/home/ubuntu/images/" --output_dir "sd-lora0fp16" \
  --hub_model_id "jlbaker361/sd-lora0fp16" \
  --train_gradient_accumulation_steps 2 \
  --train_batch_size 2 --num_epochs 2 --sample_num_steps 2  \
  --sample_batch_size 2  --sample_num_batches_per_epoch 4 --mixed_precision "fp16"
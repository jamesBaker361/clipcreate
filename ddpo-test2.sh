python3 ddpo_train_script.py --pretrained_model_name_or_path "jlbaker361/sd-wikiart-lora-0epoch" \
  --image_dir "~/vanilla-images/" --output_dir "sd-wikiart-lora-0epoch-ddpo" \
  --hub_model_id "jlbaker361/sd-wikiart-lora-0epoch-ddpo" \
  --train_gradient_accumulation_steps 4 \
  --train_batch_size 4 --num_epochs 2 --sample_num_steps 30  \
  --sample_batch_size 4  --sample_num_batches_per_epoch 8
python3 ddpo_train_script.py --image_dir "~/ddpo-dcgan-images/" --output_dir "ddpo-stability-dcgan-images" --hub_model_id "jlbaker361/ddpo-stability-dcgan" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 10 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32  --resume_from "~/clipcreate/ddpo-stability-dcgan-images/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"
python3 ddpo_train_script.py --image_dir "~/ddpo-dcgan-images/" --output_dir "ddpo-stability-dcgan-images" --hub_model_id "jlbaker361/ddpo-stability-dcgan" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 10 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32  --resume_from "~/clipcreate/ddpo-stability-dcgan-images/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"
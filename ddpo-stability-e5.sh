python3 ddpo_train_script.py --image_dir "~/ddpo-stability-e5-images/" --output_dir "ddpo-stability-e5" --hub_model_id "jlbaker361/ddpo-stability-e5" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 5 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32 --dataset_name "jlbaker361/wikiart-balanced1000"
python3 ddpo_train_script.py --image_dir "~/ddpo-stability-dcgan-e5-images/" --output_dir "ddpo-stability-dcgan-e5" --hub_model_id "jlbaker361/ddpo-stability-dcgan-e5" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 5 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32 --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"

python3 ddpo_train_script.py --image_dir "~/ddpo-stability-e8-images/" --output_dir "ddpo-stability-e8" --hub_model_id "jlbaker361/ddpo-stability-e8" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 8 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32 --dataset_name "jlbaker361/wikiart-balanced1000"
python3 ddpo_train_script.py --image_dir "~/ddpo-stability-dcgan-e8-images/" --output_dir "ddpo-stability-dcgan-e8" --hub_model_id "jlbaker361/ddpo-stability-dcgan-e8" --train_gradient_accumulation_steps 1 --train_batch_size 16 --num_epochs 8 --sample_num_steps 30  --sample_batch_size 16  --sample_num_batches_per_epoch 32 --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"

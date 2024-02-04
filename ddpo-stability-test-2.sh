#python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad-2" --hub_model_id "jlbaker361/ddpo-bad-test-2" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 2 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --pretrained_model_name_or_path "jlbaker361/ddpo-bad-test-2"
#python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad-3" --hub_model_id "jlbaker361/ddpo-bad-test-3" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 4 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --pretrained_model_name_or_path "jlbaker361/ddpo-bad-test-2"
#python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad-4" --hub_model_id "jlbaker361/ddpo-bad-test-4" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 6 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --pretrained_model_name_or_path "jlbaker361/ddpo-bad-test-3"

#python3 ddpo_train_script.py --image_dir "~/ddpo-bad-dcgan-images/" --output_dir "ddpo-bad-dcgan" --hub_model_id "jlbaker361/ddpo-bad-dcgan-test" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 2 --sample_num_steps 30  --sample_batch_size 4  --sample_num_batches_per_epoch 1  --resume_from "~/clipcreate/ddpo-bad-dcgan/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"
#python3 ddpo_train_script.py --image_dir "~/ddpo-bad-dcgan-images/" --output_dir "ddpo-bad-dcgan" --hub_model_id "jlbaker361/ddpo-bad-dcgan-test" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 4 --sample_num_steps 30  --sample_batch_size 4  --sample_num_batches_per_epoch 1  --resume_from "~/clipcreate/ddpo-bad-dcgan/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000" --reward_function dcgan --dcgan_repo_id "jlbaker361/dcgan-wikiart1000-resized"

python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad" --hub_model_id "jlbaker361/ddpo-bad-test-2" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 2 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --resume_from "~/clipcreate/ddpo-bad/checkpoints"
python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad" --hub_model_id "jlbaker361/ddpo-bad-test-2" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 4 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --resume_from "~/clipcreate/ddpo-bad/checkpoints"
python3 ddpo_train_script.py --image_dir "~/ddpo-bad-images/" --output_dir "ddpo-bad" --hub_model_id "jlbaker361/ddpo-bad-test-2" --train_gradient_accumulation_steps 1 --train_batch_size 4 --num_epochs 6 --sample_num_steps 4  --sample_batch_size 4  --sample_num_batches_per_epoch 1    --dataset_name "jlbaker361/wikiart-balanced500" --resume_from "~/clipcreate/ddpo-bad/checkpoints"
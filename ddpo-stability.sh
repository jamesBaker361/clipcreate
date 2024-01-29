python3 ddpo_train_script.py --image_dir "~/ddpo-images/" --output_dir "ddpo-stability-images" --hub_model_id "jlbaker361/ddpo-stability" --train_gradient_accumulation_steps 8 --train_batch_size 8 --num_epochs 40 --sample_num_steps 50  --sample_batch_size 8  --sample_num_batches_per_epoch 32  --resume_from "~/clipcreate/ddpo-stability/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000"
python3 ddpo_train_script.py --image_dir "~/ddpo-images/" --output_dir "ddpo-stability-images" --hub_model_id "jlbaker361/ddpo-stability" --train_gradient_accumulation_steps 8 --train_batch_size 8 --num_epochs 40 --sample_num_steps 50  --sample_batch_size 8  --sample_num_batches_per_epoch 32  --resume_from "~/clipcreate/ddpo-stability/checkpoints" --dataset_name "jlbaker361/wikiart-balanced1000" 
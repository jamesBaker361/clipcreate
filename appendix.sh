#sbatch runpygpu.sh appendix_images.py --limit 5 --can_model_list "jlbaker361/dcgan-lazy-wikiart1000-resized" "jlbaker361/dcgan-lazy-wikiart500-clip-resized" "jlbaker361/dcgan-lazy-wikiart1000-resized" --conditional True
#sbatch runpygpu.sh appendix_images.py --limit 5 --can_model_list "jlbaker361/dcgan-wikiart1000-resized" "jlbaker361/dcgan-wikiart1000-clip-resized"
sbatch runpygpu.sh appendix_images.py --limit 5 --ddpo_model_list "jlbaker361/ddpo-stability" "jlbaker361/ddpo-stability-dcgan" --conditional True --file_path "table-ddpo.png"
sbatch runpygpu.sh appendix_images.py --limit 5 --ddpo_model_list "jlbaker361/ddpo-stability" "jlbaker361/ddpo-stability-dcgan" --file_path "table-ddpo-cond.png"
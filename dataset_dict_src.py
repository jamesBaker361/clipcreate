dataset_dict={}
#"wikiart","synthetic-data","wikiart-balanced500","wikiart-mediums","wikiart-subjects"
'''
dataset_dict["wikiart"]=["jlbaker361/eval_512_wikiart-weird_dcgan_25_30",
                    "jlbaker361/eval_512_wikiart-weird_clip_25_30",
                    "jlbaker361/eval_512_wikiart-weird_kmeans_25_30",
                    "jlbaker361/eval_512_wikiart_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart_vanilla_30",
                    "jlbaker361/eval_512_wikiart_vanilla_10"]

dataset_dict["synthetic-data"]=["jlbaker361/eval_512_synthetic-data_dcgan_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_30",
                    "jlbaker361/eval_512_synthetic-data_kmeans_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_10",
                    "jlbaker361/eval_512_synthetic-data_vanilla_30",
                    "jlbaker361/eval_512_synthetic-data_vanilla_10"]

dataset_dict["wikiart-balanced500"]=["jlbaker361/eval_512_wikiart-balanced500_dcgan_30",
                    "jlbaker361/eval_512_wikiart_clip_30",
                    "jlbaker361/eval_512_wikiart-balanced500_kmeans_30",
                    "jlbaker361/eval_512_wikiart_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart-balanced500_vanilla_30",
                    "jlbaker361/eval_512_wikiart-balanced500_vanilla_10"]

dataset_dict["wikiart-mediums"]=["jlbaker361/eval_512_wikiart-mediums_dcgan_30",
                    "jlbaker361/eval_512_wikiart-mediums_clip_30",
                    "jlbaker361/eval_512_wikiart-mediums_kmeans_30",
                    "jlbaker361/eval_512_wikiart-mediums_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart-mediums_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart-mediums_vanilla_30",
                    "jlbaker361/eval_512_wikiart-mediums_vanilla_10"]

dataset_dict["wikiart-subjects"]=["jlbaker361/eval_512_wikiart-subjects_dcgan_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_30",
                    "jlbaker361/eval_512_wikiart-subjects_kmeans_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_30",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_10"]

dataset_dict["shitty"]=[
    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_30",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_10"
]

dataset_dict["everyone"]=[
    "jlbaker361/eval_512_wikiart-weird_dcgan_25_30",
                    "jlbaker361/eval_512_wikiart-weird_clip_25_30",
                    "jlbaker361/eval_512_wikiart-weird_kmeans_25_30",
                    "jlbaker361/eval_512_wikiart-mediums_dcgan_30",
                    "jlbaker361/eval_512_wikiart-mediums_clip_30",
                    "jlbaker361/eval_512_wikiart-mediums_kmeans_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_10",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_30",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_10"
]

dataset_dict["only_trained"]=[
    "jlbaker361/eval_512_wikiart-weird_dcgan_25_30",
                    "jlbaker361/eval_512_wikiart-weird_clip_25_30",
                    "jlbaker361/eval_512_wikiart-weird_kmeans_25_30",
                    "jlbaker361/eval_512_wikiart-mediums_dcgan_30",
                    "jlbaker361/eval_512_wikiart-mediums_clip_30",
                    "jlbaker361/eval_512_wikiart-mediums_kmeans_30",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30",
]
'''

dataset_dict["everyone_subjects"]=[
    "jlbaker361/eval_512_wikiart-weird_dcgan_25_30_subjects",
                    "jlbaker361/eval_512_wikiart-weird_clip_25_30_subjects",
                    "jlbaker361/eval_512_wikiart-weird_kmeans_25_30_subjects",
                    "jlbaker361/eval_512_wikiart-mediums_dcgan_30_subjects",
                    "jlbaker361/eval_512_wikiart-mediums_clip_30_subjects",
                    "jlbaker361/eval_512_wikiart-mediums_kmeans_30_subjects",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_30_subjects",
                    "jlbaker361/eval_512_wikiart-subjects_clip_prompt_10_subjects",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_30_subjects",
                    "jlbaker361/eval_512_wikiart-subjects_vanilla_10_subjects"
]



label_dict={}
label_dict["shitty"]=["M7","M8","M9"]
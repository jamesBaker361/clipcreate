DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "jlbaker361/sd-wikiart20": ("image", "text")
}
WIKIART_STYLES=[]
with open("styles.txt","r+") as file:
    WIKIART_STYLES=[f.strip() for f in file.readlines()]
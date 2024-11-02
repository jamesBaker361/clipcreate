import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--limit",type=int,default=10)
parser.add_argument("--dataset_list",nargs="*")
parser.add_argument("--size",type=int,default=100)
parser.add_argument("--name",type=str,default="grid_image")
parser.add_argument("--col_labels",nargs="*")



def create_image_grid(images, row_labels, col_labels, size, cell_size):
    # Assume each image in 'images' is already an Image object of 'cell_size'
    c, r = size
    
    # Calculate the size of the grid canvas
    label_width = 100  # Width for the row labels
    label_height = 50  # Height for the column labels
    grid_width = c * cell_size[0] + label_width
    grid_height = r * cell_size[1] + label_height
    
    # Create a blank canvas
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Load a font
    try:
        font = ImageFont.truetype("times_new_roman.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid_img)
    
    # Draw the column labels
    text_width=100
    text_height=20
    for col in range(c):
        #text_width, text_height = font.getsize(col_labels[col])
        x = col * cell_size[0] + label_width + (cell_size[0] - text_width) // 2
        y = 7  # Padding from the top
        text=col_labels[col].replace("of","of\n")
        draw.text((x, y), text, font=font, fill='black')
    
    # Draw the row labels
    for row in range(r):
        #text_width, text_height = font.getsize(row_labels[row])
        x = 10  # Padding from the left
        y = row * cell_size[1] + label_height + (cell_size[1] - text_height) // 2
        draw.text((x, y), row_labels[row], font=font, fill='black')
    
    for row in range(r):
        for col in range(c):
            img = images[col][row]
            img = img.resize(cell_size)
            
            x = col * cell_size[0] + label_width
            y = row * cell_size[1] + label_height
            
            grid_img.paste(img, (x, y))
    
    return grid_img


def main(args):
    cell_size=(args.size,args.size)
    image_matrix=[[row["image"].resize(cell_size) for row in load_dataset(f"{data}",split="train")][args.start:args.limit] for data in args.dataset_list]
    row_labels=[row["prompt"] for row in  load_dataset(args.dataset_list[0],split="train") ][args.start:args.limit]
    size=(len(args.dataset_list),args.limit-args.start)
    if args.col_labels is not None and len(args.col_labels)>0:
        col_labels=args.col_labels
    else:
        #col_labels=[f"M{x}" for x in range(len(args.dataset_list))]
        col_labels=["disc\n-full","clip\n-full","kmeans\n-full","disc\n-med","clip\n-med","kmeans\n-med","utility\n-30","utility\n-10","basic\n-30","basic\n-10"][:len(args.dataset_list)]
    grid_img=create_image_grid(image_matrix,row_labels,col_labels,size,cell_size)

    grid_img.save(f"appendix/{args.name}.png")
    

    

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")
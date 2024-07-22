#possibility space of generator

import os
import argparse
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",return_tensors="pt",do_rescale=False)
#processor.to(device)
model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)


def create_image_grid_with_labels(images, row_labels, col_labels, image_size, label_font_size=20):
    """
    Create a grid of images with axis labels.

    :param images: 2D list of PIL Image objects.
    :param row_labels: List of labels corresponding to the rows of images.
    :param col_labels: List of labels corresponding to the columns of images.
    :param image_size: Tuple (width, height) representing the size of each image.
    :param label_font_size: Font size for the labels.
    :return: PIL Image object representing the grid.
    """
    # Find unique row and column labels
    unique_row_labels = sorted(set(row_labels))
    unique_col_labels = sorted(set(col_labels))

    # Create mapping from labels to grid positions
    row_map = {label: idx for idx, label in enumerate(unique_row_labels)}
    col_map = {label: idx for idx, label in enumerate(unique_col_labels)}

    # Calculate the grid size
    num_rows = len(unique_row_labels)
    num_cols = len(unique_col_labels)

    img_width, img_height = image_size

    # Calculate the total size including labels
    label_padding = label_font_size + 10
    grid_width = num_cols * img_width + label_padding
    grid_height = num_rows * img_height + label_padding

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_img)

    # Load a default font
    font = ImageFont.load_default()

    # Place each image in the correct position based on the labels
    for i in range(len(images)):
        for j in range(len(images[i])):
            img = images[i][j]
            # Get the position from labels
            row_idx = row_map[row_labels[i]]
            col_idx = col_map[col_labels[j]]
            # Calculate the position where the image will be pasted
            x = col_idx * img_width + label_padding
            y = row_idx * img_height + label_padding
            # Resize the image if it's not the correct size
            img = img.resize(image_size)
            # Paste the image into the grid
            grid_img.paste(img, (x, y))

    # Draw labels for columns
    for label, col_idx in col_map.items():
        x = col_idx * img_width + label_padding + img_width // 2
        y = 10  # Position for the column label
        draw.text((x, y), str(label), fill='black', font=font, anchor='mm')

    # Draw labels for rows
    for label, row_idx in row_map.items():
        x = 10  # Position for the row label
        y = row_idx * img_height + label_padding + img_height // 2
        draw.text((x, y), str(label), fill='black', font=font, anchor='mm')

    return grid_img

def get_image_embeddings(image_list):
    inputs=processor(text=["wow"],images=image_list,return_tensors="pt",padding=True)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    return model(**inputs).image_embeds.cpu().detach().numpy()

parser=argparse.ArgumentParser()
parser.add_argument("--dataset_list",nargs="*",default=None)

def main(args):
    all_images=[]
    all_embeddings=[]
    labels=[]
    grid=[[None for _i in args.dataset_list] for  _j in args.dataset_list]

    embeddings_by_dataset=[]

    # Create a colormap
    colors = plt.cm.rainbow(np.linspace(0, 1, len(args.dataset_list)))

    # Map class to color
    class_to_color = {cls: color for cls, color in zip(args.dataset_list, colors)}

    for i,dataset_name in enumerate(args.dataset_list):
        dataset=load_dataset(dataset_name,split="train")
        images=[row["image"] for row in dataset]
        all_images+=images
        embeddings=np.array([get_image_embeddings([img])[0] for img in images])
        embeddings_by_dataset.append(embeddings)
        all_embeddings+=[e for e in embeddings]
        labels+=[dataset_name for _ in dataset]
        reduced_embeddings=TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=50).fit_transform(embeddings)
        x_coords = [coord[0] for coord in reduced_embeddings]
        y_coords = [coord[1] for coord in reduced_embeddings]
        plt.scatter(x_coords, y_coords, color=class_to_color[dataset_name], label=dataset_name)

        # Add titles and labels
        #plt.title('Scatter Plot of Coordinates')
        #plt.xlabel('X Coordinates')
        #plt.ylabel('Y Coordinates')

        # Add a legend
        #plt.legend()

        # Add a grid
        #plt.grid(True)

        name=dataset_name.split("/")[-1]

        # Save the plot as a PNG image
        plt.savefig(f'plots/scatter_plot_{name}.png')
        plt.savefig("temp.png")
        grid[i][i]=Image.open("temp.png")
        plt.show()
        plt.clf()
    for i in range(len(args.dataset_list)):
        for j in range(i+1, len(args.dataset_list)):
            print(i,j)
            print(args.dataset_list[i],args.dataset_list[j])
            print(len(load_dataset(args.dataset_list[i],split="train")),len(load_dataset(args.dataset_list[j],split="train")))
            embeddings=[e for e in embeddings_by_dataset[i]]+[e for e in embeddings_by_dataset[j]]
            print('len(embeddings)',len(embeddings))
            labels=[args.dataset_list[i] for _ in load_dataset(args.dataset_list[i],split="train")]+[args.dataset_list[j] for _ in load_dataset(args.dataset_list[j],split="train")]
            print('len(labels)',len(labels))
            reduced_embeddings=TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=50).fit_transform(np.array(embeddings))
            print('len(reduced_embeddings)',len(reduced_embeddings))
            x_coords = [coord[0] for coord in reduced_embeddings]
            y_coords = [coord[1] for coord in reduced_embeddings]
            unique_classes = list(set(labels))

            for cls in unique_classes:
                # Get the indices for points in the current class
                indices = [z for z, c in enumerate(labels) if c == cls]
                # Scatter plot for the current class
                plt.scatter(
                    [reduced_embeddings[z][0] for z in indices],
                    [reduced_embeddings[z][1] for z in indices],
                    color=class_to_color[cls],
                    label=cls,
                )
            plt.savefig("temp.png")
            grid[j][i]=Image.open("temp.png")
            grid[i][j]=Image.open("temp.png")
            plt.show()
            plt.clf()
    print(grid)
    all_reduced_embeddings = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=50).fit_transform(np.array(all_embeddings))
    print(all_reduced_embeddings.shape)
    x_max=max([red[0] for red in all_reduced_embeddings])
    x_min=min([red[0] for red in all_reduced_embeddings])
    y_max=max([red[1] for red in all_reduced_embeddings])
    y_min=min([red[1] for red in all_reduced_embeddings])

    print(x_max,x_min)
    print(y_max,y_min)

    unique_classes = list(set(labels))

    for cls in unique_classes:
        # Get the indices for points in the current class
        indices = [i for i, c in enumerate(labels) if c == cls]
        # Scatter plot for the current class
        plt.scatter(
            [all_reduced_embeddings[i][0] for i in indices],
            [all_reduced_embeddings[i][1] for i in indices],
            color=class_to_color[cls],
            label=cls,
        )

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot with Different Classes')

    # Add a legend
    plt.legend(title='Classes')
    plt.savefig(f'plots/scatter_plot_all.png') 

    image_grid=create_image_grid_with_labels(grid, ["a","b","c"],["a","b","c"],(250,250))
    image_grid.save("plots/grid.png")



if __name__=="__main__":

    args=parser.parse_args()
    print(args)
    start=time.time()
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")
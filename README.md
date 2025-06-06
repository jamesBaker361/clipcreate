# 🎨 clipcreate

**clipcreate** is a Python-based toolkit for training and experimenting with generative models that blend creativity and structure. It integrates KMeans clustering, GANs, and diffusion-based models to explore creative generation techniques.

## 🚀 Features

- **KMeans Clustering**: Generate cluster centers for data preprocessing.
- **Creative Adversarial Network (GAN)**: Train GANs with a focus on creativity.
- **Diffusion Model Training**: Utilize diffusion models for generative tasks.
- **Custom Loss Functions**: Implement aesthetic and creative loss functions to guide training.

## 📦 Installation

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r minimal_requirements.txt
```

## 🛠️ Usage

### 1. Generate KMeans Centers

Use `kmeans.py` to compute cluster centers:

```bash
python kmeans.py --data <"text"|"image"> --dataset <YOUR DATASET HF HUB ID>  --base_path <PATH TO SAVE CLUSTERS TO>
```

### 2. Train Creative Adversarial Network

Train the GAN model with:

```bash
python gan_training.py 
```
See the file for list of commands and options

### 3. Train Diffusion Model

Train the diffusion model using:

```bash
python ddpo_train_script.py 
```

See the file for list of commands and options

## 📁 Project Structure

- `aesthetic_reward.py`: Implements aesthetic reward functions.
- `creative_loss.py`: Defines loss functions aimed at enhancing creativity.
- `gan_training.py`: Script to train the GAN model.
- `ddpo_train_script.py`: Script to train the diffusion model.
- `kmeans.py`: Script to perform KMeans clustering.
- `minimal_requirements.txt`: List of minimal dependencies required.
- `styles.txt`: Contains style definitions or references.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

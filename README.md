# Image-Classifier

This project classifies flower species using a Vision Transformer (ViT) model from PyTorch's `torchvision. models`. The dataset of flower images is processed, trained, and used for prediction through the following workflow:

## Output 

![Top 5 Class](./top-5-class.png)

## Workflow

Dataset: The flower dataset is downloaded from a public source using wget:
```
!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
```

Training the Model: A Vision Transformer (vit_b_16) is used to classify the flower images 

```
!python train.py ./flowers --save_dir ./save_directory/checkpoint.pth --arch "vit_b_16" --learning_rate 0.001 --hidden_units 512 --epochs 30 --gpu
```
Prediction: The predict.py script uses the trained model to predict the top 5 probable classes of a test image, with category names mapped from a JSON file
```
!python predict.py ./flowers/test/10/image_07090.jpg ./save_directory/checkpoint.pth --top_k 5 --category_names ./cat_to_name.json --gpu
```
The project provides scripts for both training and prediction, utilizing GPU acceleration for efficient computation. 

## HTML View

[Image Classifier Project](./Image_Classifier_Project.html)

## NB

> **⚠️ Warning: The dataset and model checkpoint are not included in this repository.**
>
> - **Dataset:** You can download the flower dataset using [this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).
> - **Model Checkpoint:** Please provide or download the trained model checkpoint separately.

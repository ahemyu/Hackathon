# Hackathon submission

## Data preprocessing

These are the steps we implemented in the preprocessing.py : 

    Flip Image: This step horizontally flips right retina images. Since the anatomical structures in left and right retina images are mirrored, this step standardizes the orientation, ensuring consistent input to the model regardless of the eye side.

    Remove Black Background: We trim the excessive black background, focusing the model's attention on the retina itself. This reduction of irrelevant areas in the image helps in better feature extraction by the model, which is essential for accurate age prediction.

    Resize and Pad Image: Standardizing image size to 224x224 which is required by the architecture of the resnet152 model we used. Padding ensures that this resizing doesn't distort the retina's aspect ratio, maintaining the structural integrity of the image.

    Z-Normalization: This step involves standardizing the pixel intensity values across all images. By normalizing the pixel values, we reduce the variance within the dataset, allowing the model to learn more effectively from the essential features rather than being influenced by brightness or contrast differences.

    Conversion to Tensor Format: As the final step, we convert the processed images into tensors suitable for input into our deep learning model. This conversion aligns the data format with what is typically required by PyTorch-based neural networks.


## Model description

We finetuned a Resnet152 model to predict the age of the indivual based on the retina image. We implemented several methods to boost the performance of the model
(such as first freezing all of the layers expect the last one and train only the weights of that layer and then training again with all of the layers unfrozen, data Augmentation to increase the size of our dataset, hyperparam finetuning (such as learning rate, batch size, weight decay, Learning Rate Scheduler, Early stopping, ...)). We only trained on healthy individuals as we couldn't figure out how to reliably alter the prediction for individuals suffering from an illness/ilnesses. The training run of this model took approx. 16h to train 


## Installation
In order to install the environment execute: conda env create -f environment.yml
Then activate the environment with 'conda activate envir'

## Run
Download the model into the same folder as the main.py. Then run main.py with the file path in the stdin. The predicted age will be printed to the stdout. 
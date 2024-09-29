import os
import pandas as pd
import torch
from torchvision import models
from preprocessing import preprocessing
from model import predict

def load_model():
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model_weights_path = 'models/resnet152_1_healthy.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu') ))    
    model.eval()
    return model

def get_actual_age(image_name, annotations_df):
    # Extract the ID from the image name
    image_id = int(image_name.split('_')[0])
    # Find the corresponding age in the annotations DataFrame
    actual_age = annotations_df[annotations_df['ID'] == image_id]['Age'].values[0]
    return actual_age

if __name__ == "__main__":
    # Load annotations into a DataFrame
    annotations_df = pd.read_csv('data/annotations.csv')

    preprocessor = preprocessing()
    model = load_model()

    image_folder_path = 'data/Images'

       # Open a text file to write the output
    with open('age_differences_emovic.txt', 'w') as file:
        for image_name in os.listdir(image_folder_path):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_folder_path, image_name)

                normalized_image = preprocessor.preprocess(image_path)
                predicted_age = predict(model, normalized_image)

                actual_age = get_actual_age(image_name, annotations_df)

                age_difference = predicted_age - actual_age
                file.write(f"Image: {image_name} - Age Difference: {age_difference}\n")

    print("Processing complete. Age differences written to age_differences.txt")

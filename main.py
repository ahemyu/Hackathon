# Import the preprocessing class
import preprocessing
import importlib
importlib.reload(preprocessing)
from preprocessing import preprocessing
import torch
from torchvision import models
from model import predict

if __name__ == "__main__":
    preprocessor = preprocessing()
    ## load in the finetuned model
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1) # added last layer for regression

    model_weights_path = 'model.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu') ))
    model.eval()

    pathToFile = input()
    # normalize image to suitable format to be fed into the model 
    normalized_image = preprocessor.preprocess(pathToFile)
    
    res = predict(model, normalized_image)
    print(res)


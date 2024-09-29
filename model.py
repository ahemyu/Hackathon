import numpy as np


def predict(model, image):

    # Now you can use model for inference
    output = model(image)
    predicted_age = round(output.detach().numpy()[0][0].item(), 3)
    
    return predicted_age

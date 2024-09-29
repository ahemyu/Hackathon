import cv2
import numpy as np
from PIL import Image
import torch


class preprocessing():

    def flip_image(self, image_path):
        """
        Flip the image horizontally if it's a right retina image.

        Args:
        image_path (str): The path to the retina image.

        Returns:
        numpy.ndarray: The flipped image if it's a right retina, otherwise the original image.
        """
        if "right" in image_path.lower():
            # Load the image
            image = cv2.imread(image_path)

            # Check if image is loaded properly
            if image is None:
                raise ValueError(f"Image not found at {image_path}")

            # Flip the image horizontally
            flipped_image = cv2.flip(image, 1)

            return flipped_image

        else:
            # Load and return the original image if it's not a right retina image
            image = cv2.imread(image_path)
            

            # Check if image is loaded properly
            if image is None:
                raise ValueError(f"Image not found at {image_path}")

            return image
        
    def remove_black_background(self, image):
        """
        Remove as much black background as possible while keeping the retina in the image.

        Args:
        image (numpy.ndarray): The input image.

        Returns:
        numpy.ndarray: Image with minimal black background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour and its bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the image to this bounding box
            cropped_image = image[y:y+h, x:x+w]

            return cropped_image
        else:
            # Return original image if no contours found
            return image
            

    def resize_and_pad_image(self, image, target_size=(224, 224)):
        """
        Resize and pad the image to the target size.

        Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size (width, height).

        Returns:
        numpy.ndarray: Resized and padded image.
        """

        # Compute the scale and the required padding
        scale = min(target_size[0] / image.shape[1], target_size[1] / image.shape[0])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        delta_w = target_size[0] - new_size[0]
        delta_h = target_size[1] - new_size[1]

        # Resize the image
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        # Pad the resized image
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return padded_image

    def z_normalize_image(self, image):
        """
        Apply Z-normalization to the image excluding the black background.

        Args:
        image (numpy.ndarray): The input image.

        Returns:
        numpy.ndarray: Z-normalized image.
        
        """
        # Create a mask for non-black pixels (assuming black is [0, 0, 0])
        mask = image > 0

        # Initialize the normalized image
        normalized_image = np.zeros_like(image, dtype=np.float32)

        # Apply Z-normalization per channel
        for c in range(3):
            channel = image[:, :, c]
            valid_pixels = channel[mask[:, :, c]]
            
            # Avoid division by zero for images with a channel having all black/zero pixels
            if valid_pixels.size == 0:
                continue

            mean = valid_pixels.mean()
            std = valid_pixels.std()

            # Avoid division by zero if std is 0 (e.g., a solid color channel)
            if std > 0:
                normalized_channel = (channel - mean) / std
                normalized_image[:, :, c] = normalized_channel * mask[:, :, c]

        # convert to tensor to be fed into model
        image_tensor = torch.from_numpy(normalized_image).float()
        if image_tensor.dim() == 3:  # if image is in format [H, W, C]
            image_tensor = image_tensor.permute(2, 0, 1)  # convert to [C, H, W]
        # add a batch dimension if not already present (required by underlying resnet model)
        if image_tensor.dim() == 3:  # [C, H, W]
            image_tensor = image_tensor.unsqueeze(0)  

        normalized_image = image_tensor

        return normalized_image

    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline for a given image path.

        Args:
        image_path (str): Path to the image.

        Returns:
        numpy.ndarray: Preprocessed image.
        """
        image = self.flip_image(image_path)
        image = self.remove_black_background(image)
        image = self.resize_and_pad_image(image)
        image = self.z_normalize_image(image)

        return image
    

    





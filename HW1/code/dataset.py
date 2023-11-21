import os
import cv2
import numpy as np
from PIL import Image


def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    dataset = []
    for filename in os.listdir(dataPath+"/face"):
        file = cv2.imread(os.path.join(dataPath + "/face", filename), 0)
        # file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        # new_file = np.squeeze(file)
        if file is not None:
            image = (file, 1)
            dataset.append(image)

    for filename in os.listdir(dataPath+"/non-face"):
        file = cv2.imread(os.path.join(dataPath + "/non-face", filename), 0)
        # file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        if file is not None:
            image = (file, 0)
            dataset.append(image)
    return dataset
    # End your code (Part 1)
    

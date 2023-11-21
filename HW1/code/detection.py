import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    with open(dataPath) as dtxt:
        image = np.zeros(shape=[512, 512], dtype=np.uint8)
        imgname = ""
        for line in dtxt.read().splitlines():
            lst = line.split(" ")
            if len(lst) == 2:
                imgname = lst[0]
                for root, dirs, files in os.walk(str(Path(dataPath).parent)):
                  if imgname in files:
                    image = cv2.imread(os.path.join(root, imgname))
                    # print("FOUND! Image Shape: ", image.shape)
            elif len(lst) == 4:
                x, y, w, h = int(lst[0]), int(lst[1]), int(lst[2]), int(lst[3])
                startpoint = (x, y)
                endpoint = (x+w, y+h)
                thickness = 3
                crop = image[y:y+h, x:x+w].copy()
                face = cv2.resize(crop, (19, 19))
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if clf.classify(gray):
                   color = (0, 255, 0)
                  #  print("Is Face!")
                else:
                   color = (0, 0, 255)
                  #  print("Not face!")
                image = cv2.rectangle(image, startpoint, endpoint, color, thickness)
                cv2.imshow(imgname, image)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
    dtxt.close()
    # End your code (Part 4)

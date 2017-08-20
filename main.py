import numpy as np
import os
import datetime
from PIL import Image
import glob

timeStarted = datetime.datetime.utcnow()

imagesPathsArray = glob.glob("coil-100" + os.path.sep + "*.png")

allImagesArray = []

for imagePath in imagesPathsArray:
    image = Image.open(imagePath)
    rgbMatrix = np.asarray(image)
    imageVector = rgbMatrix.reshape(128*128, -1)
    allImagesArray.append(imageVector)

print(allImagesArray)
print(len(allImagesArray))

timeEnded = datetime.datetime.utcnow()
print("Rodou em: " + str((timeEnded - timeStarted).total_seconds()))

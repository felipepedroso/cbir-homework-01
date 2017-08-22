import numpy as np
import os
import datetime
from PIL import Image
import glob
import cv2
from sklearn.cluster import KMeans

timeStarted = datetime.datetime.utcnow()

imagesPathsArray = glob.glob("coil-100" + os.path.sep + "*.png")

imagesHistograms = {}

for imagePath in imagesPathsArray:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=5)
    clt.fit(image)

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (imageHistogram, _) = np.histogram(clt.labels_, bins=numLabels)
    imageHistogram = imageHistogram.astype("float")
    imageHistogram /= imageHistogram.sum()

    imagesHistograms[imagePath] = imageHistogram


timeEnded = datetime.datetime.utcnow()
print("Rodou em: " + str((timeEnded - timeStarted).total_seconds()))

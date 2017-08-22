import numpy as np
import os
import datetime
from PIL import Image
import glob
from sklearn.cluster import KMeans

timeStarted = datetime.datetime.utcnow()

#imagesPathsArray = glob.glob("coil-100" + os.path.sep + "*.png")

imagesPathsArray = np.empty(0)
for i in range(1,11):
    filename = "coil-100\\obj" + str(i) + "__0.png"
    imagesPathsArray = np.append(imagesPathsArray, filename)


image = Image.open("coil-100\obj42__0.png")
rgbMatrix = np.asarray(image)
imageVector = rgbMatrix.reshape(128*128, -1)
centroids_a = KMeans(n_clusters=5).fit(imageVector).cluster_centers_

total_distances = np.empty(shape=(0,2))
i = 0
for imagePath in imagesPathsArray:
    i += 1
    print("Processing " + imagePath)
    image = Image.open(imagePath)
    rgbMatrix = np.asarray(image)
    imageVector = rgbMatrix.reshape(128*128, -1)
    centroids_b = KMeans(n_clusters=5).fit(imageVector).cluster_centers_

    distances = np.empty(0)

    for c in range(0,5):
        soma = 0
        for rgb in range(0,3):
            soma += (centroids_b[c][rgb]-centroids_a[c][rgb])**2
        distances = np.append(distances, soma**(1/2))
    distancia_and_index = str(distances.sum())+ "+" + str(i)
    total_distances = np.append(total_distances, distancia_and_index)

print(total_distances)

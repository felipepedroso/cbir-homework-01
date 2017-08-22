# USAGE
# python compare.py --dataset images

# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import datetime

timeStarted = datetime.datetime.utcnow()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory of images")
ap.add_argument("-k", "--keyimage", required = True, help = "Key image filename")
ap.add_argument("-m", "--method", required = True, help = "Distance computation method")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}
count = 0

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*.png"):
        count += 1
        print("Loading image: " + str(count))
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	index[filename] = hist

# UTILIZING SCIPY
# initialize the scipy methods to compute distances
SCIPY_METHODS = (
	("Euclidean", dist.euclidean),
	("Manhattan", dist.cityblock),
	("Chebysev", dist.chebyshev))
methodName, method = SCIPY_METHODS[int(args["method"])]

# initialize the dictionary dictionary
results = {}
count = 0

# loop over the index
for (k, hist) in index.items():
        count += 1
        print("Computing distance of image: " + str(count))
        # compute the distance between the two histograms
	# using the method and update the results dictionary
	d = method(index[args["dataset"]+"\\"+args["keyimage"]], hist)
	results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])

# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images[args["dataset"]+"\\"+args["keyimage"]])
plt.axis("off")

# initialize the results figure
fig = plt.figure("Results: %s" % (methodName))
fig.suptitle(methodName, fontsize = 20)

# loop over the results
for (i, (v, k)) in enumerate(results[:20]):
	# show the result
	#ax = fig.add_subplot(1, len(images), i + 1)
	ax = fig.add_subplot(1, 20, i + 1)
	#ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")

# show the SciPy methods

timeEnded = datetime.datetime.utcnow()
print("Finished in: " + str((timeEnded - timeStarted).total_seconds()))

plt.show()

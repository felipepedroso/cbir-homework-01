
# coding: utf-8

# In[4]:


import numpy as np
import os
import datetime
from PIL import Image
import glob
import struct
import scipy
import scipy.misc
import scipy.cluster


# In[5]:


timeStarted = datetime.datetime.utcnow()

imagesPathsArray = glob.glob("coil-100" + os.path.sep + "*.png")

allImagesArray = []
kmeansCenterArray = []


# In[6]:


for imagePath in imagesPathsArray:
    image = Image.open(imagePath) #open image
    rgbMatrix = np.asarray(image) #convert to RGB
    imageVector = rgbMatrix.reshape(128*128, -1) 
    print(imageVector)
    allImagesArray.append(imageVector)


# In[13]:


count = 0
while (count < len(allImagesArray)):
   
    codes, dist = scipy.cluster.vq.kmeans(allImagesArray[count]*1.0,5)
    kmeansCenterArray.append(codes)
    count = count + 1
    
    print(count)


# In[14]:


print (kmeansCenterArray[0])


# In[15]:


print (kmeansCenterArray[1])


# In[16]:


print (kmeansCenterArray[7199])


# In[ ]:





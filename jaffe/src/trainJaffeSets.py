from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as mathdude
import cv2
import pickle
import os

from loadJaffeDataset import loadJaffeImages, jaffeDatasetDir, jaffeRootDir


def preProcessImages(images, targetSize=(48, 48)):
    processedImages = []
    for img in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, targetSize)
        normalizedImg = img / 255.0
        processedImages.append(normalizedImg)
    return mathdude.array(processedImages)

#i hate python
jaffeImages, jaffeLabels = loadJaffeImages(jaffeDatasetDir)
jaffeProcessed = preProcessImages(jaffeImages)

labelEncoder = LabelEncoder()
encodedJaffeLabels = labelEncoder.fit_transform(jaffeLabels)

xTrain, xTest, yTrain, yTest = train_test_split(jaffeProcessed, encodedJaffeLabels, test_size=0.2, random_state=42, stratify=encodedJaffeLabels)



outputDir = os.path.join(jaffeRootDir, "results", "jaffeTrain")
os.makedirs(outputDir, exist_ok=True)

mathdude.save(os.path.join(outputDir, 'xTrain.npy'), xTrain)
mathdude.save(os.path.join(outputDir, 'xTest.npy'), xTest)
mathdude.save(os.path.join(outputDir, 'yTrain.npy'), yTrain)
mathdude.save(os.path.join(outputDir, 'yTest.npy'), yTest)

with open(os.path.join(outputDir, 'labelEncoder.pkl'), 'wb') as f:
    pickle.dump(labelEncoder, f)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as mathdude
import cv2
import pickle
import os

from loadCKDataset import loadCkImages, ckDatasetDir, ckRootDir


def preProcessImages(images):
    processedImages = []
    for img in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalizedImg = img / 255.0
        processedImages.append(normalizedImg)
    return mathdude.array(processedImages)

#i hate python
ckImages, ckLabels = loadCkImages(ckDatasetDir)
ckProcessed = preProcessImages(ckImages)

labelEncoder = LabelEncoder()
encodedCkLabels = labelEncoder.fit_transform(ckLabels)

xTrain, Xtrest, yTrain, yTest = train_test_split(ckProcessed, encodedCkLabels, test_size=0.2, random_state=42, stratify=encodedCkLabels)



outputDir = os.path.join(ckRootDir, "results", "ckTrain")
os.makedirs(outputDir, exist_ok=True)

mathdude.save(os.path.join(outputDir, 'xTrain.npy'), xTrain)
mathdude.save(os.path.join(outputDir, 'xTest.npy'), Xtrest)
mathdude.save(os.path.join(outputDir, 'yTrain.npy'), yTrain)
mathdude.save(os.path.join(outputDir, 'yTest.npy'), yTest)

with open(os.path.join(outputDir, 'labelEncoder.pkl'), 'wb') as f:
    pickle.dump(labelEncoder, f)
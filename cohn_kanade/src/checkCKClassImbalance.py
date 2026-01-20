from sklearn.preprocessing import LabelEncoder
from collections import Counter
from loadCKDataset import loadCkImages, ckDatasetDir
import numpy as mathdude

ckImages, ckLabels = loadCkImages(ckDatasetDir)
labelCounts = Counter(ckLabels)

labelEncoder = LabelEncoder()
encodedLabels = labelEncoder.fit_transform(ckLabels)

unique, counts = mathdude.unique(encodedLabels, return_counts=True)

for label, count in zip(unique, counts): #zip combines two lists together to make an array or something
    emotionName = labelEncoder.inverse_transform([label])[0]
    percentage = (count / len(ckLabels)) * 100
    print(f"Emotion: {emotionName}, Count: {count}, Percentage: {percentage:.2f}%")
    
maxCount = max(counts)
minCount = min(counts)
imbalanceRatio = maxCount / minCount 
    
print(f"Class Imbalance Ratio (Max/Min): {imbalanceRatio:.2f}")
if imbalanceRatio > 1.5:
       print("IMBALANCED")
else:
    print("Balanced")

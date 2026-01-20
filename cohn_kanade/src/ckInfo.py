import os
import cv2
import matplotlib.pyplot as graphguy
from collections import Counter 

from loadCKDataset import loadCkImages, ckDatasetDir,ckRootDir

ckImages, ckLabels = loadCkImages(ckDatasetDir)


labelCounts = Counter(ckLabels)
print(labelCounts)

graphguy.figure(figsize=(10,5))
emotions = list(labelCounts.keys())
counts = list(labelCounts.values())
graphguy.bar(emotions, counts, color='skyblue')
graphguy.xlabel('Emotion')
graphguy.ylabel('Number of Images')
graphguy.title('CK Dataset - Emotion Distribution')
graphguy.xticks(rotation=45)
graphguy.tight_layout()

os.makedirs(os.path.join(ckRootDir, 'results', 'ckInfo'), exist_ok=True)
graphguy.savefig(os.path.join(ckRootDir, 'results', 'ckInfo', 'ckDistribution.png'))


graphguy.figure(figsize=(15,3))
emotions = set()
plotIndex = 1

for i, label in enumerate(ckLabels):
    if label not in emotions:
        graphguy.subplot(1,6,plotIndex)
        imgRgb = cv2.cvtColor(ckImages[i], cv2.COLOR_BGR2RGB)
        graphguy.imshow(imgRgb)
        graphguy.title(label)
        graphguy.axis('off')
        emotions.add(label)
        plotIndex += 1
    if len(emotions) >= 6:
        break
graphguy.savefig(os.path.join(ckRootDir, 'results', 'ckInfo', 'ckSampleImages.png'))

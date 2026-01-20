import os
import cv2
import matplotlib.pyplot as graphguy
from collections import Counter

from loadJaffeDataset import loadJaffeImages, jaffeDatasetDir, jaffeRootDir

jaffeImages, jaffeLabels = loadJaffeImages(jaffeDatasetDir)


labelCounts = Counter(jaffeLabels)
print(labelCounts)

graphguy.figure(figsize=(10,5))
emotions = list(labelCounts.keys())
counts = list(labelCounts.values())
graphguy.bar(emotions, counts, color='skyblue')
graphguy.xlabel('Emotion')
graphguy.ylabel('Number of Images')
graphguy.title('JAFFE Dataset - Emotion Distribution')
graphguy.xticks(rotation=45)
graphguy.tight_layout()

os.makedirs(os.path.join(jaffeRootDir, 'results', 'jaffeInfo'), exist_ok=True)
graphguy.savefig(os.path.join(jaffeRootDir, 'results', 'jaffeInfo', 'jaffeDistribution.png'))


graphguy.figure(figsize=(15,3))
emotions = set()
plotIndex = 1

for i, label in enumerate(jaffeLabels):
    if label not in emotions:
        graphguy.subplot(1,7,plotIndex)
        imgRgb = cv2.cvtColor(jaffeImages[i], cv2.COLOR_BGR2RGB)
        graphguy.imshow(imgRgb)
        graphguy.title(label)
        graphguy.axis('off')
        emotions.add(label)
        plotIndex += 1
    if len(emotions) >= 7:
        break
graphguy.savefig(os.path.join(jaffeRootDir, 'results', 'jaffeInfo', 'jaffeSampleImages.png'))

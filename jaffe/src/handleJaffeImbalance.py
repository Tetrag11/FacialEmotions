from imblearn.over_sampling import SMOTE #for smiting the league of legends objectives, also for handling class imbalances
from collections import Counter
import numpy as mathdude
import pickle
import os

from loadJaffeDataset import loadJaffeImages, jaffeDatasetDir, jaffeRootDir

def loadTrainingData():
   inputDir = os.path.join(jaffeRootDir, "results", "jaffeTrain")
   xTrain = mathdude.load(os.path.join(inputDir, 'xTrain.npy'))
   yTrain = mathdude.load(os.path.join(inputDir, 'yTrain.npy'))
   with open(os.path.join(inputDir, 'labelEncoder.pkl'), 'rb') as f:
         labelEncoder = pickle.load(f)
   return xTrain, yTrain, labelEncoder


def applySmote(xTrain, yTrain, randomState=42):
    nSamples, height, width = xTrain.shape #shape returns the number of items, their height and width
    #convert to 2d
    xTrain2d = xTrain.reshape(nSamples, -1)

    #smite the dragon
    smote = SMOTE(random_state=randomState)
    xResampled2D, yResampled = smote.fit_resample(xTrain2d, yTrain)

    xResampled  = xResampled2D.reshape(-1, height, width)
    return xResampled, yResampled


def printDistribution(y, labelEncoder, title):
    counts = Counter(y)
    print(f"\n{title}")
    print("-" * 40)
    for emotionIdx, count in sorted(counts.items()):
        emotionLabel = labelEncoder.inverse_transform([emotionIdx])[0]
        print(f"{emotionLabel:10s}: {count}")
    print(f" Total: {len(y)} images")


def main():
    xTrain, yTrain, labelEncoder = loadTrainingData()
    printDistribution(yTrain, labelEncoder, "Before SMOTE:")

    countsBefore = Counter(yTrain)
    maxCount = max(countsBefore.values())
    minCount = min(countsBefore.values())
    imbalanceRatio = maxCount / minCount
    print(f"\nImbalance ratio (max/min): {imbalanceRatio:.2f}")

    xTrainBalanced, yTrainBalanced = applySmote(xTrain, yTrain)
    printDistribution(yTrainBalanced, labelEncoder, "After SMOTE:")

    countsAfter = Counter(yTrainBalanced)
    maxCountAfter = max(countsAfter.values())
    minCountAfter = min(countsAfter.values())
    imbalanceRatioAfter = maxCountAfter / minCountAfter
    print(f"\nImbalance ratio after SMOTE: {imbalanceRatioAfter:.2f}")

    outputDir = os.path.join(jaffeRootDir, "results", "jaffeTrainBalanced")
    inputDir = os.path.join(jaffeRootDir, "results", "jaffeTrain")
    os.makedirs(outputDir, exist_ok=True)

    mathdude.save(os.path.join(outputDir, 'xTrainBalanced.npy'), xTrainBalanced)
    mathdude.save(os.path.join(outputDir, 'yTrainBalanced.npy'), yTrainBalanced)

    xTest = mathdude.load(os.path.join(inputDir, 'xTest.npy'))
    yTest = mathdude.load(os.path.join(inputDir, 'yTest.npy'))
    mathdude.save(os.path.join(outputDir, 'xTest.npy'), xTest)
    mathdude.save(os.path.join(outputDir, 'yTest.npy'), yTest)

    with open(os.path.join(outputDir, 'labelEncoder.pkl'), 'wb') as f:
        pickle.dump(labelEncoder, f)
    print(f"Data saved successfully ")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Original training samples: {len(yTrain)}")
    print(f"Balanced training samples: {len(yTrainBalanced)}")
    print(f"Synthetic samples added: {len(yTrainBalanced) - len(yTrain)}")

if __name__ == "__main__":
    main()

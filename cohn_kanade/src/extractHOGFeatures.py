from skimage.feature import hog 
import numpy as mathdude
import pickle
import os

from loadCKDataset import ckRootDir


def loadBalancedData():
    inputDir = os.path.join(ckRootDir, 'results', 'ckTrainBalanced')
    xTrain = mathdude.load(os.path.join(inputDir, 'xTrainBalanced.npy'))
    xTest = mathdude.load(os.path.join(inputDir, 'xTest.npy'))
    yTrain = mathdude.load(os.path.join(inputDir, 'yTrainBalanced.npy'))
    yTest = mathdude.load(os.path.join(inputDir, 'yTest.npy'))
    
    with open(os.path.join(inputDir, 'labelEncoder.pkl'), 'rb') as f:
        labelEncoder = pickle.load(f)
    return xTrain, xTest, yTrain, yTest, labelEncoder



def extractHogFeatures(images, orientations=9, pixelsPerCell=(8, 8), cellsPerBlock=(2, 2)):
    featuresList = []
    
    for i, image in enumerate(images):
         featureVector = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixelsPerCell,
            cells_per_block=cellsPerBlock,
            visualize=False,
            feature_vector=True
        )
         featuresList.append(featureVector)
    
    return mathdude.array(featuresList) 


def main():
    xTrain, xTest, yTrain, yTest, labelEncoder = loadBalancedData()
    
    print(f"Training images shape: {xTrain.shape}")
    print(f"Test images shape: {xTest.shape}")
    
    xTrainHog = extractHogFeatures(xTrain)
    xTestHog = extractHogFeatures(xTest)
    
    print(f"\nTraining HOG features shape: {xTrainHog.shape}")
    print(f"Test HOG features shape: {xTestHog.shape}")
    
    outputDir = os.path.join(ckRootDir, 'results', 'ckTrainBalancedHOG')
    os.makedirs(outputDir, exist_ok=True)
    
    mathdude.save(os.path.join(outputDir, 'xTrainHog.npy'), xTrainHog)
    mathdude.save(os.path.join(outputDir, 'xTestHog.npy'), xTestHog)
    mathdude.save(os.path.join(outputDir, 'yTrain.npy'), yTrain)
    mathdude.save(os.path.join(outputDir, 'yTest.npy'), yTest)
    
    with open(os.path.join(outputDir, 'labelEncoder.pkl'), 'wb') as f:
        pickle.dump(labelEncoder, f)
        
    print(f"\nHOG features saved to {outputDir}")

if __name__ == "__main__":
    main()
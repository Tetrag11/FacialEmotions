from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as mathdude
import pickle
import os
from sklearn.decomposition import PCA #it compresses the amount of features my recoginizing the most important patterns, using this for consistency with jaffe pipeline
from loadCKDataset import ckRootDir
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def loadHogFeatures():
    inputDir = os.path.join(ckRootDir, 'results', 'ckTrainBalancedHOG')
    xTrain = mathdude.load(os.path.join(inputDir, 'xTrainHog.npy'))
    xTest = mathdude.load(os.path.join(inputDir, 'xTestHog.npy'))
    yTrain = mathdude.load(os.path.join(inputDir, 'yTrain.npy'))
    yTest = mathdude.load(os.path.join(inputDir, 'yTest.npy'))

    with open(os.path.join(inputDir, 'labelEncoder.pkl'), 'rb') as f:
        labelEncoder = pickle.load(f)

    return xTrain, xTest, yTrain, yTest, labelEncoder

def trainSvm(xtrain, ytrain):
    pipe = Pipeline([('pca', PCA(random_state=42)), ('svm', SVC(kernel='rbf', class_weight="balanced",random_state=42))])
    param_grid = {
        'pca__n_components': [20, 30, 42, 50, 75, 100],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.01, 0.1, 1]
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', verbose=2)
    grid.fit(xtrain, ytrain)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.2%}")
    return grid.best_estimator_

def evaluateModel(model, xtrain, ytrain, xtest, ytest, labelEncoder):
    yTrainPred = model.predict(xtrain)
    yTestPred = model.predict(xtest)
    
    trainAccuracy = accuracy_score(ytrain, yTrainPred)
    testAccuracy = accuracy_score(ytest, yTestPred)
    
    emotionLabels = labelEncoder.classes_
    
    print(f"\nTraining Accuracy: {trainAccuracy*100:.2f}%")
    print(f"Test Accuracy: {testAccuracy*100:.2f}%")
    
    print("\nClassification Report (Test Set)")
    print(classification_report(ytest, yTestPred, target_names=emotionLabels))

    
    return trainAccuracy, testAccuracy

def saveModel(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"SVM model saved to {path}")
    
def main():
    xTrain, xTest, yTrain, yTest, labelEncoder = loadHogFeatures()
    
    print(f"Training features shape: {xTrain.shape}")
    print(f"Test features shape: {xTest.shape}")
    
    svmModel = trainSvm(xTrain, yTrain)
    
    trainAcc, testAcc = evaluateModel(svmModel, xTrain, yTrain, xTest, yTest, labelEncoder)

    
    outputDir = os.path.join(ckRootDir, 'results', 'ck_svm_model')
    os.makedirs(outputDir, exist_ok=True)
    saveModel(svmModel, os.path.join(outputDir, 'svm_model.pkl'))

    print(f"\nFinal Training accuracy: {trainAcc*100:.2f}%")
    print(f"Final Test accuracy: {testAcc*100:.2f}%")
    
if __name__ == "__main__":
    main()
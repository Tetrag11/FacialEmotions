from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as graphguy
import seaborn as heatmapguy
import numpy as mathdude
import pickle
import os

from loadJaffeDataset import jaffeRootDir


def loadModelAndData():
    hogDir = os.path.join(jaffeRootDir, 'results', 'jaffeTrainBalancedHOG')
    balancedDir = os.path.join(jaffeRootDir, 'results', 'jaffeTrainBalanced')
    modelDir = os.path.join(jaffeRootDir, 'results', 'jaffe_svm_model')

    xTrainHog = mathdude.load(os.path.join(hogDir, 'xTrainHog.npy'))
    xTestHog = mathdude.load(os.path.join(hogDir, 'xTestHog.npy'))
    yTrain = mathdude.load(os.path.join(hogDir, 'yTrain.npy'))
    yTest = mathdude.load(os.path.join(hogDir, 'yTest.npy'))

    xTestImages = mathdude.load(os.path.join(balancedDir, 'xTest.npy'))

    with open(os.path.join(modelDir, 'svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(hogDir, 'labelEncoder.pkl'), 'rb') as f:
        labelEncoder = pickle.load(f)

    return xTrainHog, xTestHog, yTrain, yTest, xTestImages, model, labelEncoder

def plotConfusionMatrix(yTrue, yPred, labelEncoder, savePath):
    cm = confusion_matrix(yTrue, yPred)
    emotionNames = labelEncoder.classes_

    graphguy.figure(figsize=(10, 8))
    heatmapguy.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotionNames,
        yticklabels=emotionNames
    )
    graphguy.title('Confusion Matrix - JAFFE Dataset\nSVM with HOG Features')
    graphguy.xlabel('Predicted Emotion')
    graphguy.ylabel('Actual Emotion')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()


def plotAccuracyComparison(trainAcc, testAcc, savePath):
    graphguy.figure(figsize=(8, 6))

    bars = graphguy.bar(
        ['Training', 'Testing'],
        [trainAcc * 100, testAcc * 100],
        color=['#2ecc71', '#3498db'],
        edgecolor='black'
    )

    for bar, acc in zip(bars, [trainAcc, testAcc]):
        graphguy.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc * 100:.1f}%',
            ha='center',
            fontweight='bold'
        )

    graphguy.ylim(0, 110)
    graphguy.ylabel('Accuracy (%)')
    graphguy.title('Training vs Testing Accuracy\nSVM with HOG Features on JAFFE')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()


def plotPerClassAccuracy(yTrue, yPred, labelEncoder, savePath):
    emotionNames = labelEncoder.classes_
    nClasses = len(emotionNames)

    classAccuracies = []
    for i in range(nClasses):
        mask = yTrue == i
        if mask.sum() > 0:
            acc = (yPred[mask] == i).sum() / mask.sum()
        else:
            acc = 0
        classAccuracies.append(acc * 100)

    colors = graphguy.cm.RdYlGn([acc / 100 for acc in classAccuracies])

    graphguy.figure(figsize=(10, 6))
    bars = graphguy.bar(emotionNames, classAccuracies, color=colors, edgecolor='black')

    for bar, acc in zip(bars, classAccuracies):
        graphguy.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc:.0f}%',
            ha='center',
            fontweight='bold'
        )

    graphguy.ylim(0, 110)
    graphguy.ylabel('Accuracy (%)')
    graphguy.xlabel('Emotion')
    graphguy.title('Per-Class Accuracy on Test Set\nSVM with HOG Features on JAFFE')
    graphguy.xticks(rotation=45, ha='right')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()


def plotSamplePredictions(xTestImages, xTestHog, yTest, model, labelEncoder, savePath, nSamples=6):
    yPred = model.predict(xTestHog)
    emotionNames = labelEncoder.classes_

    selectedIndices = []
    for emotionIdx in range(len(emotionNames)):
        indices = mathdude.where(yTest == emotionIdx)[0]
        if len(indices) > 0:
            selectedIndices.append(indices[0])
        if len(selectedIndices) >= nSamples:
            break

    fig, axes = graphguy.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, idx in enumerate(selectedIndices[:6]):
        ax = axes[i]
        ax.imshow(xTestImages[idx], cmap='gray')
        ax.axis('off')

        actual = emotionNames[yTest[idx]]
        predicted = emotionNames[yPred[idx]]

        color = 'green' if actual == predicted else 'red'
        symbol = '✓' if actual == predicted else '✗'

        ax.set_title(f'Actual: {actual}\nPredicted: {predicted} {symbol}', color=color)

    graphguy.suptitle('Sample Predictions on Test Images')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()


def generateMetricsReport(yTrain, yTrainPred, yTest, yTestPred, labelEncoder, savePath):
    emotionNames = labelEncoder.classes_

    with open(savePath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("JAFFE Facial Emotion Recognition - Evaluation Report\n")
        f.write("Model: SVM with RBF Kernel + HOG Features\n")
        f.write("=" * 60 + "\n\n")

        trainAcc = accuracy_score(yTrain, yTrainPred)
        testAcc = accuracy_score(yTest, yTestPred)

        f.write("OVERALL ACCURACY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Accuracy: {trainAcc*100:.2f}%\n")
        f.write(f"Testing Accuracy:  {testAcc*100:.2f}%\n\n")

        f.write("CLASSIFICATION REPORT (Test Set)\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(yTest, yTestPred, target_names=emotionNames))
        f.write("\n")

        f.write("CONFUSION MATRIX (Test Set)\n")
        f.write("-" * 40 + "\n")
        cm = confusion_matrix(yTest, yTestPred)
        f.write("\nRows: Actual, Columns: Predicted\n\n")

        header = " " * 12 + "".join([e[:6].center(8) for e in emotionNames])
        f.write(header + "\n")
        for i, emotion in enumerate(emotionNames):
            row = emotion[:10].ljust(12) + "".join([str(cm[i, j]).center(8) for j in range(len(emotionNames))])
            f.write(row + "\n")


def main():
    outputDir = os.path.join(jaffeRootDir, 'results', 'jaffe_evaluation')
    os.makedirs(outputDir, exist_ok=True)

    print("Loading model and data...")
    xTrainHog, xTestHog, yTrain, yTest, xTestImages, model, labelEncoder = loadModelAndData()

    print("Generating predictions...")
    yTrainPred = model.predict(xTrainHog)
    yTestPred = model.predict(xTestHog)

    trainAcc = accuracy_score(yTrain, yTrainPred)
    testAcc = accuracy_score(yTest, yTestPred)

    print(f"\nTraining Accuracy: {trainAcc*100:.2f}%")
    print(f"Test Accuracy: {testAcc*100:.2f}%")

    print("\n1. Confusion Matrix...")
    plotConfusionMatrix(yTest, yTestPred, labelEncoder, f'{outputDir}/confusion_matrix.png')

    print("2. Accuracy Comparison...")
    plotAccuracyComparison(trainAcc, testAcc, f'{outputDir}/accuracy_comparison.png')

    print("3. Per-Class Accuracy...")
    plotPerClassAccuracy(yTest, yTestPred, labelEncoder, f'{outputDir}/per_class_accuracy.png')

    print("4. Sample Predictions...")
    plotSamplePredictions(xTestImages, xTestHog, yTest, model, labelEncoder, f'{outputDir}/sample_predictions.png')

    print("5. Text Report...")
    generateMetricsReport(yTrain, yTrainPred, yTest, yTestPred, labelEncoder, f'{outputDir}/evaluation_report.txt')

    print(f"\nAll files saved to: {outputDir}/")


if __name__ == "__main__":
    main()

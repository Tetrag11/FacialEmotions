import matplotlib.pyplot as graphguy
import numpy as mathdude
import pickle
import os
import re

# root directories
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jaffeRoot = os.path.join(projectRoot, 'jaffe')
ckRoot = os.path.join(projectRoot, 'cohn_kanade')
outputDir = os.path.join(projectRoot, 'comparison', 'results')


def loadEvaluationData(datasetRoot, datasetName):
    """Load evaluation results for a dataset"""
    if datasetName == 'jaffe':
        evalDir = os.path.join(datasetRoot, 'results', 'jaffe_evaluation')
        hogDir = os.path.join(datasetRoot, 'results', 'jaffeTrainBalancedHOG')
        modelDir = os.path.join(datasetRoot, 'results', 'jaffe_svm_model')
    else:
        evalDir = os.path.join(datasetRoot, 'results', 'ck_evaluation')
        hogDir = os.path.join(datasetRoot, 'results', 'ckTrainBalancedHOG')
        modelDir = os.path.join(datasetRoot, 'results', 'ck_svm_model')

    # load test data and model
    xTestHog = mathdude.load(os.path.join(hogDir, 'xTestHog.npy'))
    yTest = mathdude.load(os.path.join(hogDir, 'yTest.npy'))

    with open(os.path.join(hogDir, 'labelEncoder.pkl'), 'rb') as f:
        labelEncoder = pickle.load(f)

    with open(os.path.join(modelDir, 'svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    # load training data for train accuracy
    xTrainHog = mathdude.load(os.path.join(hogDir, 'xTrainHog.npy'))
    yTrain = mathdude.load(os.path.join(hogDir, 'yTrain.npy'))

    return {
        'xTrainHog': xTrainHog,
        'xTestHog': xTestHog,
        'yTrain': yTrain,
        'yTest': yTest,
        'labelEncoder': labelEncoder,
        'model': model
    }


def calculateAccuracies(data):
    """Calculate train and test accuracies"""
    model = data['model']

    yTrainPred = model.predict(data['xTrainHog'])
    yTestPred = model.predict(data['xTestHog'])

    trainAcc = (yTrainPred == data['yTrain']).mean()
    testAcc = (yTestPred == data['yTest']).mean()

    return trainAcc, testAcc, yTestPred


def calculatePerClassAccuracy(yTrue, yPred, labelEncoder):
    """Calculate accuracy for each emotion class"""
    emotionNames = labelEncoder.classes_
    accuracies = {}

    for i, emotion in enumerate(emotionNames):
        mask = yTrue == i
        if mask.sum() > 0:
            acc = (yPred[mask] == i).sum() / mask.sum()
        else:
            acc = 0
        accuracies[emotion] = acc * 100

    return accuracies


def plotAccuracyComparison(jaffeTrainAcc, jaffeTestAcc, ckTrainAcc, ckTestAcc, savePath):
    """Side-by-side bar chart comparing JAFFE and CK+ accuracies"""
    graphguy.figure(figsize=(10, 6))

    x = mathdude.arange(2)
    width = 0.35

    jaffeAccs = [jaffeTrainAcc * 100, jaffeTestAcc * 100]
    ckAccs = [ckTrainAcc * 100, ckTestAcc * 100]

    bars1 = graphguy.bar(x - width/2, jaffeAccs, width, label='JAFFE', color='#e74c3c', edgecolor='black')
    bars2 = graphguy.bar(x + width/2, ckAccs, width, label='CK+', color='#3498db', edgecolor='black')

    # add value labels on bars
    for bar in bars1:
        graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{bar.get_height():.1f}%', ha='center', fontweight='bold', fontsize=10)
    for bar in bars2:
        graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{bar.get_height():.1f}%', ha='center', fontweight='bold', fontsize=10)

    graphguy.xlabel('Dataset Split', fontsize=12)
    graphguy.ylabel('Accuracy (%)', fontsize=12)
    graphguy.title('JAFFE vs CK+ Dataset Accuracy Comparison\nSVM with HOG Features', fontsize=14)
    graphguy.xticks(x, ['Training', 'Testing'])
    graphguy.ylim(0, 115)
    graphguy.legend(loc='upper right')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()
    print(f"Saved: {savePath}")


def plotPerClassComparison(jaffePerClass, ckPerClass, savePath):
    """Compare per-class accuracy between datasets"""
    # get common emotions (both datasets)
    commonEmotions = sorted(set(jaffePerClass.keys()) & set(ckPerClass.keys()))

    graphguy.figure(figsize=(12, 6))

    x = mathdude.arange(len(commonEmotions))
    width = 0.35

    jaffeAccs = [jaffePerClass.get(e, 0) for e in commonEmotions]
    ckAccs = [ckPerClass.get(e, 0) for e in commonEmotions]

    bars1 = graphguy.bar(x - width/2, jaffeAccs, width, label='JAFFE', color='#e74c3c', edgecolor='black')
    bars2 = graphguy.bar(x + width/2, ckAccs, width, label='CK+', color='#3498db', edgecolor='black')

    # add value labels
    for bar in bars1:
        if bar.get_height() > 0:
            graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{bar.get_height():.0f}%', ha='center', fontsize=9)
    for bar in bars2:
        if bar.get_height() > 0:
            graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{bar.get_height():.0f}%', ha='center', fontsize=9)

    graphguy.xlabel('Emotion', fontsize=12)
    graphguy.ylabel('Accuracy (%)', fontsize=12)
    graphguy.title('Per-Emotion Accuracy Comparison\nJAFFE vs CK+ on Test Set', fontsize=14)
    graphguy.xticks(x, commonEmotions, rotation=45, ha='right')
    graphguy.ylim(0, 115)
    graphguy.legend(loc='upper right')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()
    print(f"Saved: {savePath}")


def plotDatasetSizeComparison(jaffeData, ckData, savePath):
    """Compare dataset sizes"""
    graphguy.figure(figsize=(10, 6))

    x = mathdude.arange(2)
    width = 0.35

    jaffeSizes = [len(jaffeData['yTrain']), len(jaffeData['yTest'])]
    ckSizes = [len(ckData['yTrain']), len(ckData['yTest'])]

    bars1 = graphguy.bar(x - width/2, jaffeSizes, width, label='JAFFE', color='#e74c3c', edgecolor='black')
    bars2 = graphguy.bar(x + width/2, ckSizes, width, label='CK+', color='#3498db', edgecolor='black')

    for bar in bars1:
        graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{int(bar.get_height())}', ha='center', fontweight='bold')
    for bar in bars2:
        graphguy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{int(bar.get_height())}', ha='center', fontweight='bold')

    graphguy.xlabel('Dataset Split', fontsize=12)
    graphguy.ylabel('Number of Samples', fontsize=12)
    graphguy.title('Dataset Size Comparison\nJAFFE vs CK+', fontsize=14)
    graphguy.xticks(x, ['Training (Balanced)', 'Testing'])
    graphguy.legend(loc='upper right')
    graphguy.tight_layout()
    graphguy.savefig(savePath, dpi=150)
    graphguy.close()
    print(f"Saved: {savePath}")


def generateComparisonReport(jaffeData, ckData, jaffeTrainAcc, jaffeTestAcc, ckTrainAcc, ckTestAcc,
                             jaffePerClass, ckPerClass, savePath):
    """Generate text comparison report"""
    with open(savePath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET COMPARISON REPORT: JAFFE vs CK+\n")
        f.write("Facial Emotion Recognition using SVM with HOG Features\n")
        f.write("=" * 70 + "\n\n")

        # dataset sizes
        f.write("DATASET SIZES\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Dataset':<15} {'Train (Balanced)':<20} {'Test':<15}\n")
        f.write(f"{'JAFFE':<15} {len(jaffeData['yTrain']):<20} {len(jaffeData['yTest']):<15}\n")
        f.write(f"{'CK+':<15} {len(ckData['yTrain']):<20} {len(ckData['yTest']):<15}\n\n")

        # overall accuracy
        f.write("OVERALL ACCURACY\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Dataset':<15} {'Training':<15} {'Testing':<15} {'Gap':<15}\n")
        jaffeGap = (jaffeTrainAcc - jaffeTestAcc) * 100
        ckGap = (ckTrainAcc - ckTestAcc) * 100
        f.write(f"{'JAFFE':<15} {jaffeTrainAcc*100:<15.2f} {jaffeTestAcc*100:<15.2f} {jaffeGap:<15.2f}\n")
        f.write(f"{'CK+':<15} {ckTrainAcc*100:<15.2f} {ckTestAcc*100:<15.2f} {ckGap:<15.2f}\n\n")

        # per-class accuracy
        f.write("PER-CLASS TEST ACCURACY (%)\n")
        f.write("-" * 50 + "\n")
        commonEmotions = sorted(set(jaffePerClass.keys()) & set(ckPerClass.keys()))
        f.write(f"{'Emotion':<15} {'JAFFE':<15} {'CK+':<15} {'Difference':<15}\n")
        for emotion in commonEmotions:
            jAcc = jaffePerClass.get(emotion, 0)
            cAcc = ckPerClass.get(emotion, 0)
            diff = cAcc - jAcc
            f.write(f"{emotion:<15} {jAcc:<15.1f} {cAcc:<15.1f} {diff:+<15.1f}\n")

        # emotions only in one dataset
        jaffeOnly = set(jaffePerClass.keys()) - set(ckPerClass.keys())
        ckOnly = set(ckPerClass.keys()) - set(jaffePerClass.keys())

        if jaffeOnly:
            f.write(f"\nJAFFE only emotions: {', '.join(jaffeOnly)}\n")
        if ckOnly:
            f.write(f"CK+ only emotions: {', '.join(ckOnly)}\n")

        f.write("\n")
        f.write("KEY OBSERVATIONS\n")
        f.write("-" * 50 + "\n")
        f.write(f"1. CK+ has {len(ckData['yTrain'])/len(jaffeData['yTrain']):.1f}x more training samples than JAFFE\n")
        f.write(f"2. JAFFE shows {'overfitting' if jaffeGap > 20 else 'moderate generalization'} (train-test gap: {jaffeGap:.1f}%)\n")
        f.write(f"3. CK+ shows {'overfitting' if ckGap > 20 else 'good generalization'} (train-test gap: {ckGap:.1f}%)\n")
        f.write(f"4. CK+ outperforms JAFFE by {(ckTestAcc - jaffeTestAcc)*100:.1f}% on test accuracy\n")

    print(f"Saved: {savePath}")


def main():
    os.makedirs(outputDir, exist_ok=True)

    print("Loading JAFFE evaluation data...")
    jaffeData = loadEvaluationData(jaffeRoot, 'jaffe')

    print("Loading CK+ evaluation data...")
    ckData = loadEvaluationData(ckRoot, 'ck')

    print("\nCalculating accuracies...")
    jaffeTrainAcc, jaffeTestAcc, jaffeTestPred = calculateAccuracies(jaffeData)
    ckTrainAcc, ckTestAcc, ckTestPred = calculateAccuracies(ckData)

    print(f"JAFFE - Train: {jaffeTrainAcc*100:.2f}%, Test: {jaffeTestAcc*100:.2f}%")
    print(f"CK+   - Train: {ckTrainAcc*100:.2f}%, Test: {ckTestAcc*100:.2f}%")

    # per-class accuracy
    jaffePerClass = calculatePerClassAccuracy(jaffeData['yTest'], jaffeTestPred, jaffeData['labelEncoder'])
    ckPerClass = calculatePerClassAccuracy(ckData['yTest'], ckTestPred, ckData['labelEncoder'])

    print("\nGenerating comparison visualizations...")

    print("1. Accuracy comparison chart...")
    plotAccuracyComparison(jaffeTrainAcc, jaffeTestAcc, ckTrainAcc, ckTestAcc,
                          os.path.join(outputDir, 'accuracy_comparison.png'))

    print("2. Per-class comparison chart...")
    plotPerClassComparison(jaffePerClass, ckPerClass,
                          os.path.join(outputDir, 'per_class_comparison.png'))

    print("3. Dataset size comparison...")
    plotDatasetSizeComparison(jaffeData, ckData,
                             os.path.join(outputDir, 'dataset_size_comparison.png'))

    print("4. Comparison report...")
    generateComparisonReport(jaffeData, ckData, jaffeTrainAcc, jaffeTestAcc, ckTrainAcc, ckTestAcc,
                            jaffePerClass, ckPerClass,
                            os.path.join(outputDir, 'comparison_report.txt'))

    print(f"\nAll comparison files saved to: {outputDir}/")


if __name__ == "__main__":
    main()

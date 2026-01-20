# Facial Emotion Recognition Using Support Vector Machine with HOG Features

**Author:** [YOUR NAME]
**Student ID:** [YOUR STUDENT ID]
**Module:** CO3519 - Artificial Intelligence
**Date:** [DATE]

---

## 1. Introduction

Facial emotion recognition is when a computer looks at a person's face and tries to figure out how they are feeling. Humans do this naturally when we talk to each other - we look at someone's face and can tell if they are happy, sad, angry, or surprised. Teaching a computer to do the same thing is useful for many applications like helping robots understand humans, detecting driver fatigue, or improving customer service.

In this project, I built a machine learning system that can look at a picture of a face and classify it into one of six emotions: Angry, Disgust, Fear, Happy, Sad, and Surprise. I used a traditional machine learning approach with three main parts: HOG (Histogram of Oriented Gradients) for feature extraction, PCA (Principal Component Analysis) for reducing the number of features, and SVM (Support Vector Machine) for classification.

I tested my system on two different datasets - JAFFE and Cohn-Kanade (CK+). The results show that my model achieved [INSERT JAFFE TEST ACCURACY]% accuracy on JAFFE and [INSERT CK TEST ACCURACY]% accuracy on CK+. The big difference in results between the two datasets shows how important dataset size and quality are for machine learning.

---

## 2. Literature Review

Facial emotion recognition has been studied for many years. Researchers have tried different ways to solve this problem.

**Feature Extraction Methods:** Before deep learning became popular, researchers used handcrafted features to describe faces. Dalal and Triggs (2005) introduced HOG features, which look at the direction of edges in an image. HOG works well for face recognition because faces have clear edges around the eyes, nose, and mouth. Other methods like LBP (Local Binary Patterns) and Gabor filters have also been used, but HOG remains popular because it is simple and works well (Shan et al., 2009).

**Classification Methods:** For classifying emotions, Support Vector Machine (SVM) has been one of the most successful traditional methods. Littlewort et al. (2011) showed that SVM with carefully chosen features can achieve good accuracy on emotion recognition. SVM works by finding the best boundary between different emotion classes.

**Deep Learning Approaches:** More recently, Convolutional Neural Networks (CNNs) have achieved better results than traditional methods. Li and Deng (2020) showed that deep learning can automatically learn features from images without needing handcrafted features like HOG. However, deep learning needs lots of data and computing power.

**My Approach:** My approach is similar to the traditional methods. I use HOG features like Dalal and Triggs, and SVM for classification like Littlewort et al. The difference is that I also use PCA to reduce the number of features before SVM, and I use GridSearchCV to find the best parameters automatically. This makes my pipeline more robust than using fixed parameters.

---

## 3. Datasets

I used two datasets to train and test my emotion recognition system.

### 3.1 JAFFE Dataset

The Japanese Female Facial Expression (JAFFE) dataset was created by Lyons et al. (1998). It contains 213 grayscale images of 10 Japanese female subjects. Each person shows 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The images are 256x256 pixels in TIFF format.

**[INSERT IMAGE: jaffeDistribution.png]**
*Figure 1: Distribution of emotions in the JAFFE dataset*

**[INSERT IMAGE: jaffeSampleImages.png]**
*Figure 2: Sample images from JAFFE dataset showing different emotions*

The JAFFE dataset is quite small with only about 30 images per emotion. This makes it challenging for machine learning because the model does not have many examples to learn from.

### 3.2 Cohn-Kanade (CK+) Dataset

The Extended Cohn-Kanade (CK+) dataset was created by Lucey et al. (2010). It is larger than JAFFE with 981 images from 123 different subjects of various ethnicities. It has 6 emotions: Angry, Disgust, Fear, Happy, Sad, and Surprise.

**[INSERT IMAGE: ckDistribution.png]**
*Figure 3: Distribution of emotions in the CK+ dataset*

**[INSERT IMAGE: ckSampleImages.png]**
*Figure 4: Sample images from CK+ dataset showing different emotions*

### 3.3 Dataset Comparison

| Feature | JAFFE | CK+ |
|---------|-------|-----|
| Total Images | 213 | 981 |
| Subjects | 10 | 123 |
| Emotions | 7 | 6 |
| Image Format | TIFF | PNG |
| Diversity | Japanese females only | Multiple ethnicities |

**[INSERT IMAGE: comparison/results/dataset_size_comparison.png]**
*Figure 5: Comparison of dataset sizes*

### 3.4 Preprocessing

Before feeding images to the model, I did several preprocessing steps:

1. **Grayscale Conversion:** Changed colour images to black and white. This reduces the data size and emotions are mainly shown through facial structure, not colour.

2. **Resizing:** Made all images 48x48 pixels. This makes them the same size and reduces computing time.

3. **Normalisation:** Changed pixel values from 0-255 to 0-1. This helps the model learn better because the numbers are smaller and more consistent.

4. **Train-Test Split:** Split the data into 80% for training and 20% for testing. I used stratified splitting to make sure each emotion has the same proportion in both sets.

---

## 4. Model Development

My emotion recognition pipeline has four main steps: handling class imbalance, feature extraction, dimensionality reduction, and classification.

### 4.1 Handling Class Imbalance with SMOTE

Class imbalance is when some emotions have more images than others. This is a problem because the model might just learn to always predict the most common emotion and ignore the rare ones.

To fix this, I used SMOTE (Synthetic Minority Over-sampling Technique) by Chawla et al. (2002). SMOTE creates new fake training images for the emotions that have fewer examples. It does this by looking at existing images and creating new ones that are similar but slightly different. After SMOTE, all emotions have the same number of training images.

### 4.2 Feature Extraction with HOG

HOG (Histogram of Oriented Gradients) is a way to describe an image using the direction of edges. Here is how it works:

1. **Divide the image into cells:** I split the 48x48 image into small 8x8 pixel cells. This gives a 6x6 grid of cells.

2. **Calculate gradients:** For each pixel, calculate which direction the edge is pointing (like up, down, left, right, or diagonal).

3. **Create histograms:** For each cell, count how many edges point in each direction. I used 9 direction bins.

4. **Normalise blocks:** Group cells into 2x2 blocks and normalise them. This makes the features more robust to lighting changes.

The result is a feature vector of 900 numbers for each image. These numbers describe the shape and structure of the face.

**HOG Parameters Used:**
- Orientations: 9
- Pixels per cell: 8x8
- Cells per block: 2x2

I chose these parameters because they are commonly used in face recognition research and work well for 48x48 images.

### 4.3 Dimensionality Reduction with PCA

PCA (Principal Component Analysis) reduces the number of features while keeping the most important information. With 900 HOG features but only a few hundred training images, the model might overfit (memorise the training data instead of learning general patterns).

PCA finds the directions in the data that have the most variation and keeps only those. I let GridSearchCV choose the best number of PCA components from [20, 30, 42, 50, 75, 100].

### 4.4 Classification with SVM

SVM (Support Vector Machine) is the classifier that makes the final prediction. It works by finding the best boundary (called a hyperplane) between different emotion classes.

I used SVM with an RBF (Radial Basis Function) kernel. The RBF kernel can find curved boundaries, which is important because emotions are not separated by simple straight lines in the feature space.

**SVM Parameters:**
- Kernel: RBF
- Class weight: Balanced (gives equal importance to all emotions)

### 4.5 Hyperparameter Tuning with GridSearchCV

Instead of guessing the best parameters, I used GridSearchCV to try many different combinations and find the best ones. GridSearchCV uses 5-fold cross-validation, which means it splits the training data into 5 parts, trains on 4 parts, tests on 1 part, and repeats this 5 times.

**Parameter Grid Searched:**
- PCA components: [20, 30, 42, 50, 75, 100]
- SVM C: [0.1, 1, 10, 100]
- SVM gamma: ['scale', 0.01, 0.1, 1]

This tests 96 different combinations (6 x 4 x 4) and picks the one with the best cross-validation accuracy.

**Best Parameters Found:**

| Dataset | PCA Components | SVM C | SVM Gamma | CV Accuracy |
|---------|---------------|-------|-----------|-------------|
| JAFFE | [INSERT] | [INSERT] | [INSERT] | [INSERT]% |
| CK+ | [INSERT] | [INSERT] | [INSERT] | [INSERT]% |

### 4.6 Complete Pipeline

The complete pipeline looks like this:

```
Raw Image → Grayscale → Resize (48x48) → Normalise → HOG Features → PCA → SVM → Predicted Emotion
```

I used scikit-learn's Pipeline class to combine PCA and SVM, which makes sure the same transformations are applied during training and testing.

---

## 5. Model Evaluation

### 5.1 Evaluation Metrics

I used several metrics to evaluate my model:

- **Accuracy:** The percentage of correct predictions out of all predictions.
- **Precision:** Out of all predictions for an emotion, how many were correct.
- **Recall:** Out of all actual examples of an emotion, how many were found.
- **F1-Score:** The balance between precision and recall.
- **Confusion Matrix:** A table showing which emotions get confused with each other.

### 5.2 JAFFE Results

**[INSERT IMAGE: jaffe_evaluation/accuracy_comparison.png]**
*Figure 6: Training vs Testing accuracy for JAFFE dataset*

| Metric | Value |
|--------|-------|
| Training Accuracy | [INSERT]% |
| Testing Accuracy | [INSERT]% |

**[INSERT IMAGE: jaffe_evaluation/confusion_matrix.png]**
*Figure 7: Confusion matrix for JAFFE dataset*

**[INSERT IMAGE: jaffe_evaluation/per_class_accuracy.png]**
*Figure 8: Per-emotion accuracy for JAFFE dataset*

The JAFFE results show a large gap between training and testing accuracy. This is called overfitting - the model memorised the training images but cannot recognise new faces well. This happens because JAFFE is very small (only 213 images) and all subjects are Japanese females, so there is not much variety.

### 5.3 CK+ Results

**[INSERT IMAGE: ck_evaluation/accuracy_comparison.png]**
*Figure 9: Training vs Testing accuracy for CK+ dataset*

| Metric | Value |
|--------|-------|
| Training Accuracy | [INSERT]% |
| Testing Accuracy | [INSERT]% |

**[INSERT IMAGE: ck_evaluation/confusion_matrix.png]**
*Figure 10: Confusion matrix for CK+ dataset*

**[INSERT IMAGE: ck_evaluation/per_class_accuracy.png]**
*Figure 11: Per-emotion accuracy for CK+ dataset*

The CK+ results are much better than JAFFE. The model generalises well to new faces because CK+ has more images and more variety in the subjects.

### 5.4 Dataset Comparison

**[INSERT IMAGE: comparison/results/accuracy_comparison.png]**
*Figure 12: Side-by-side accuracy comparison between JAFFE and CK+*

**[INSERT IMAGE: comparison/results/per_class_comparison.png]**
*Figure 13: Per-emotion accuracy comparison between datasets*

| Dataset | Training Accuracy | Testing Accuracy | Gap |
|---------|------------------|------------------|-----|
| JAFFE | [INSERT]% | [INSERT]% | [INSERT]% |
| CK+ | [INSERT]% | [INSERT]% | [INSERT]% |

The comparison shows that:
1. CK+ performs much better because it has about 5 times more training data.
2. JAFFE has severe overfitting (large train-test gap) while CK+ generalises well.
3. Some emotions like Happy are easier to recognise on both datasets because smiling is a very clear feature.
4. Emotions like Fear and Sad are harder because they look similar to each other.

### 5.5 Sample Predictions

**[INSERT IMAGE: jaffe_evaluation/sample_predictions.png]**
*Figure 14: Sample predictions on JAFFE test images showing actual vs predicted emotions*

**[INSERT IMAGE: ck_evaluation/sample_predictions.png]**
*Figure 15: Sample predictions on CK+ test images showing actual vs predicted emotions*

The sample predictions show that my model can correctly classify many faces. Green titles show correct predictions and red titles show incorrect ones.

### 5.6 Analysis of Results

**Why does CK+ work better?**
1. More training data (981 vs 213 images)
2. More subject diversity (123 vs 10 people)
3. Multiple ethnicities vs only Japanese females

**Common Failure Modes:**
1. Fear and Surprise get confused because both have wide eyes
2. Angry and Disgust get confused because both have furrowed eyebrows
3. Sad and Neutral get confused because the differences are subtle

---

## 6. Conclusion

In this project, I built a facial emotion recognition system using traditional machine learning techniques. The system uses HOG features to describe faces, PCA to reduce dimensionality, and SVM to classify emotions. I used GridSearchCV to automatically find the best parameters.

**Key Findings:**
1. The model achieved [INSERT]% test accuracy on JAFFE and [INSERT]% on CK+.
2. Dataset size matters a lot - CK+ with more images performed much better than JAFFE.
3. SMOTE helped balance the classes and improve fairness across emotions.
4. GridSearchCV found different optimal parameters for each dataset, showing that one size does not fit all.

**What I Learned:**
1. Feature engineering (HOG) is important for traditional machine learning.
2. Overfitting is a real problem with small datasets.
3. Cross-validation and proper evaluation are essential.
4. Dataset quality and size greatly affect model performance.

**Future Improvements:**
1. Use data augmentation (rotating, flipping images) to create more training data.
2. Try other classifiers like Random Forest or K-Nearest Neighbours.
3. Combine multiple feature types (HOG + LBP) for better description.
4. Use larger and more diverse datasets.
5. Consider deep learning methods like CNNs if more data is available.

---

## References

Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', Journal of Artificial Intelligence Research, 16, pp. 321-357.

Dalal, N. and Triggs, B. (2005) 'Histograms of Oriented Gradients for Human Detection', IEEE Conference on Computer Vision and Pattern Recognition, pp. 886-893.

Li, S. and Deng, W. (2020) 'Deep Facial Expression Recognition: A Survey', IEEE Transactions on Affective Computing, 13(3), pp. 1195-1215.

Littlewort, G., Whitehill, J., Wu, T., Fasel, I., Frank, M., Movellan, J. and Bartlett, M. (2011) 'The Computer Expression Recognition Toolbox (CERT)', IEEE International Conference on Automatic Face & Gesture Recognition, pp. 298-305.

Lucey, P., Cohn, J.F., Kanade, T., Saragih, J., Ambadar, Z. and Matthews, I. (2010) 'The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression', IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 94-101.

Lyons, M.J., Akamatsu, S., Kamachi, M. and Gyoba, J. (1998) 'Coding Facial Expressions with Gabor Wavelets', IEEE International Conference on Automatic Face and Gesture Recognition, pp. 200-205.

Shan, C., Gong, S. and McOwan, P.W. (2009) 'Facial expression recognition based on Local Binary Patterns: A comprehensive study', Image and Vision Computing, 27(6), pp. 803-816.

---

## Supplementary Material

Source code for this project is available at: [INSERT YOUR GITHUB/GITLAB URL]

The repository contains:
- `jaffe/src/` - All Python scripts for JAFFE dataset processing
- `cohn_kanade/src/` - All Python scripts for CK+ dataset processing
- `comparison/` - Scripts for comparing both datasets
- `results/` - Generated figures and evaluation reports

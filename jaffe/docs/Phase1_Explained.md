# Understanding Facial Emotion Recognition: Phase 1 Explained

## What Are We Building?

Imagine you're teaching a robot to understand how people feel by looking at their faces. When you're happy, you smile. When you're sad, you might frown. We're building a computer program that can look at a photo of a face and say "this person looks happy" or "this person looks angry".

This is called **Facial Emotion Recognition** - teaching a computer to recognize emotions from faces.

---

## The Big Picture: How Does a Computer "Learn"?

Computers don't actually think or feel. They're really good at one thing: **finding patterns in numbers**.

Here's the trick: **images are just numbers**. Every photo is made up of tiny squares called **pixels**, and each pixel has a number representing how bright it is (0 = black, 255 = white).

So when we "teach" a computer to recognize emotions, we're really saying:
> "When the numbers in a face image look like THIS pattern, the person is probably happy. When they look like THAT pattern, they're probably angry."

---

## Phase 1: Preparing the Data

Before we can teach our computer anything, we need to prepare our "teaching materials". This is called **Data Preparation**.

Think of it like preparing flashcards before studying:
1. Get the flashcards (download images)
2. Look through them to understand what you have (explore the data)
3. Organize them into "study pile" and "quiz pile" (train/test split)
4. Make sure you have enough of each type (handle class imbalance)

---

## Step 1: Loading the Images

### What is a Dataset?

A **dataset** is a collection of examples that we use to teach the computer. Our dataset is called **JAFFE** (Japanese Female Facial Expression) - it contains 213 photos of Japanese women making different facial expressions.

### The Code

In [loadJaffeDataset.py](../src/loadJaffeDataset.py), we load all the images:

```python
# This dictionary maps short codes to emotion names
# In the JAFFE dataset, each image filename contains a code like "HA" for Happy
emotion_mapping = {
    'HA': 'Happy',
    'SA': 'Sad',
    'SU': 'Surprise',
    'AN': 'Angry',
    'DI': 'Disgust',
    'FE': 'Fear',
    'NE': 'Neutral'
}
```

**What's happening here?**

The JAFFE images have filenames like `KM.HA3.39.tiff`. The "HA" part tells us this is a "Happy" face. We use a **dictionary** (like a real dictionary that translates words) to convert these codes into full emotion names.

### What is an Image to a Computer?

When you load an image, the computer sees it as a grid of numbers:

```
Image: 256 x 256 pixels

[  45,  47,  52,  48, ... ]   <- Row 1: 256 numbers
[  43,  44,  50,  51, ... ]   <- Row 2: 256 numbers
[  ... 256 rows total ...  ]
```

Each number (0-255) represents brightness:
- **0** = Pure black
- **255** = Pure white
- **128** = Medium gray

A 256x256 image has 65,536 pixels - that's 65,536 numbers the computer needs to analyze!

---

## Step 2: Exploring the Data

### Why Explore?

Before studying, you'd want to know: How many flashcards do I have? Are they all the same difficulty? Do I have enough of each topic?

In [jaffeInfo.py](../src/jaffeInfo.py), we answer similar questions about our images.

### Class Distribution

A **class** in machine learning means a category. Our classes are the 7 emotions: Happy, Sad, Angry, etc.

**Class distribution** means: how many examples do we have of each emotion?

```
JAFFE Dataset Distribution:
  Happy:    31 images
  Sad:      31 images
  Surprise: 30 images
  Angry:    30 images
  Disgust:  29 images
  Fear:     32 images
  Neutral:  30 images
  -----------------
  Total:    213 images
```

This tells us we have roughly 30 images per emotion - pretty balanced!

### Why Does This Matter?

Imagine you're studying for a test with flashcards:
- 100 cards about dogs
- 2 cards about cats

You'd get really good at recognizing dogs, but terrible at cats! The same thing happens with computers - they learn better when they have similar amounts of each type.

---

## Step 3: Preprocessing Images

### What is Preprocessing?

**Preprocessing** means preparing the raw data so it's easier for the computer to learn from. It's like cleaning and organizing your study materials.

In [TrainJaffeSets.py](../src/TrainJaffeSets.py), we do three things:

### 3a. Convert to Grayscale

**Grayscale** means black-and-white (no colors).

```python
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Why?**
- Color images have 3 "channels" (Red, Green, Blue) = 3x more numbers
- For emotions, the SHAPE of the face matters more than the color
- Grayscale = simpler = faster learning

**Before (Color):**
```
Each pixel = [R, G, B] = [143, 112, 95]  (3 numbers)
256 x 256 x 3 = 196,608 numbers per image
```

**After (Grayscale):**
```
Each pixel = 117  (1 number representing brightness)
256 x 256 = 65,536 numbers per image
```

### 3b. Resize to 48x48

```python
img = cv2.resize(img, (48, 48))
```

**Why 48x48?**

Original images might be different sizes (256x256, 300x200, etc.). Computers need consistent input - like how a form needs all answers in the same size boxes.

48x48 is small enough to process quickly, but big enough to keep important facial features.

**Before:** 256 x 256 = 65,536 numbers
**After:** 48 x 48 = 2,304 numbers

That's 28x less data to process!

### 3c. Normalize to 0-1

```python
img = img / 255.0
```

**What is Normalization?**

Original pixel values range from 0 to 255. We divide by 255 to get values between 0 and 1.

**Before:** `[0, 45, 128, 255, 200, ...]`
**After:** `[0.0, 0.176, 0.502, 1.0, 0.784, ...]`

**Why?**

Machine learning algorithms work better with small numbers. It's like converting:
- Kilometers to meters
- Dollars to cents

The information is the same, just in a more convenient scale. This helps the computer:
1. Learn faster
2. Avoid numerical problems with very large numbers
3. Treat all features equally (no feature dominates just because it has bigger numbers)

---

## Step 4: Train/Test Split

### The Golden Rule of Machine Learning

> **Never test yourself with the same questions you studied!**

If you memorize the exact answers to practice questions, you might ace the practice test but fail the real exam. You need to test on NEW questions to see if you truly understand.

### Training Set vs Test Set

We split our 213 images into two groups:

```python
X_train, X_test, y_train, y_test = train_test_split(
    jaffe_processed,        # The images (X)
    jaffe_labels_encoded,   # The emotion labels (y)
    test_size=0.2,          # 20% for testing
    random_state=42,        # For reproducibility
    stratify=jaffe_labels_encoded
)
```

- **Training Set (80%):** 170 images - used to teach the computer
- **Test Set (20%):** 43 images - used to quiz the computer AFTER learning

### What Do X and y Mean?

In machine learning, we use these letters by convention:
- **X** = the input data (images) - what we show the computer
- **y** = the labels (emotions) - the correct answers

It's like flashcards:
- **X** = the front of the card (the question/image)
- **y** = the back of the card (the answer/emotion)

### What is Stratified Splitting?

```python
stratify=jaffe_labels_encoded
```

**Stratified** means keeping the same proportion of each emotion in both sets.

**Without stratification (random split):**
```
Training: 25 Happy, 28 Sad, 22 Angry, 30 Fear, 20 Disgust...
Test:     6 Happy, 3 Sad, 8 Angry, 2 Fear, 9 Disgust...
```
The test set might accidentally have very few of some emotions!

**With stratification:**
```
Training: 25 Happy, 25 Sad, 24 Angry, 25 Fear, 23 Disgust... (14% each)
Test:     6 Happy, 6 Sad, 6 Angry, 7 Fear, 6 Disgust...     (14% each)
```
Both sets have the same percentage of each emotion - fair testing!

### What is random_state=42?

```python
random_state=42
```

Splitting is random - run it twice, get different splits. But for science, we want **reproducibility** - if someone else runs our code, they should get the exact same results.

Setting `random_state=42` is like saying "use this specific random pattern". The number 42 is arbitrary (it's a joke from "The Hitchhiker's Guide to the Galaxy" - the answer to life, the universe, and everything).

---

## Step 5: Label Encoding

### The Problem

Computers don't understand words. They only understand numbers.

Our labels are: `['Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Neutral', 'Surprise']`

We need to convert these to numbers.

### The Solution: Label Encoder

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
jaffe_labels_encoded = label_encoder.fit_transform(jaffe_labels)
```

**Before:** `['Happy', 'Sad', 'Happy', 'Angry', 'Fear', ...]`
**After:** `[3, 5, 3, 0, 2, ...]`

The mapping:
```
Angry    -> 0
Disgust  -> 1
Fear     -> 2
Happy    -> 3
Neutral  -> 4
Sad      -> 5
Surprise -> 6
```

(Alphabetical order)

### Why Save the Label Encoder?

```python
with open('results/jaffe_train/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

Later, when our model predicts "3", we need to convert it back to "Happy". The label encoder remembers the mapping, so we save it for later use.

**Pickle** is Python's way of saving objects to a file - like taking a snapshot of the encoder so we can reload it later.

---

## Step 6: Handling Class Imbalance with SMOTE

### What is Class Imbalance?

Even though JAFFE is relatively balanced, imagine if we had:
```
Happy:  100 images
Sad:    100 images
Angry:   10 images  <- Problem!
```

The computer would see Angry faces only 10/(100+100+10) = 4.7% of the time. It might learn to just guess "Happy" or "Sad" and ignore "Angry" completely!

### What is SMOTE?

**SMOTE** = Synthetic Minority Over-sampling Technique

In simple terms: SMOTE creates **fake but realistic** examples of the minority classes.

### How Does SMOTE Work?

Imagine you have 3 photos of angry faces (A, B, C). SMOTE creates new angry faces by **blending** existing ones:

```
Original faces (as numbers):
Face A: [0.2, 0.5, 0.3, 0.8, ...]
Face B: [0.3, 0.4, 0.5, 0.7, ...]

SMOTE creates a new face by picking a point BETWEEN them:
New Face: [0.25, 0.45, 0.4, 0.75, ...]  (average of A and B)
```

It's like mixing two paint colors to get a new shade - the new face is a blend of two real faces.

### The Code

In [handleJaffeImbalance.py](../src/handleJaffeImbalance.py):

```python
from imblearn.over_sampling import SMOTE

# Flatten images: SMOTE needs 2D input (samples x features)
# Our images are 48x48 = 2304 features per image
X_train_flat = X_train.reshape(n_samples, -1)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled_flat, y_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back to images
X_resampled = X_resampled_flat.reshape(-1, 48, 48)
```

### Why Flatten?

SMOTE expects data in a flat format:
```
Before (3D): [170 images, 48 rows, 48 columns]
After (2D):  [170 images, 2304 features]
```

Each image becomes a single row of 2,304 numbers. After SMOTE creates new samples, we reshape back to images.

### Our Results

```
Before SMOTE:
  Angry:    24 images
  Disgust:  23 images  <- Minority
  Fear:     25 images
  Happy:    25 images
  Neutral:  24 images
  Sad:      25 images
  Surprise: 24 images
  Total:    170 images

After SMOTE:
  All classes: 25 images each
  Total: 175 images (+5 synthetic)
```

SMOTE created 5 synthetic images to make all classes have exactly 25 samples.

---

## Summary: What We've Done So Far

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA PREPARATION                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: LOAD DATA
├── Read 213 face images from JAFFE dataset
├── Extract emotion labels from filenames
└── Result: Raw images + labels

Step 2: EXPLORE DATA
├── Count images per emotion class
├── Visualize sample images
└── Result: Understanding of our dataset

Step 3: PREPROCESS
├── Convert to grayscale (3 channels → 1 channel)
├── Resize to 48x48 (consistent size, less data)
├── Normalize to 0-1 (better for learning)
└── Result: Clean, consistent images

Step 4: TRAIN/TEST SPLIT
├── 80% training (170 images)
├── 20% testing (43 images)
├── Stratified (same class proportions)
└── Result: Separate sets for learning and evaluation

Step 5: ENCODE LABELS
├── Convert text labels to numbers
├── Save encoder for later use
└── Result: Computer-readable labels

Step 6: HANDLE IMBALANCE (SMOTE)
├── Create synthetic samples for minority classes
├── Balance all classes to same count
└── Result: Fair representation of all emotions
```

---

## Files We Created

| File | Purpose |
|------|---------|
| `src/loadJaffeDataset.py` | Load images and labels from disk |
| `src/jaffeInfo.py` | Explore and visualize the dataset |
| `src/checkJaffeClassImbalance.py` | Check if classes are balanced |
| `src/TrainJaffeSets.py` | Preprocess and split the data |
| `src/handleJaffeImbalance.py` | Apply SMOTE to balance classes |

## Data We Saved

| File | Contents |
|------|----------|
| `results/jaffe_train_balanced/X_train_balanced.npy` | 175 balanced training images (48x48) |
| `results/jaffe_train_balanced/y_train_balanced.npy` | 175 training labels (numbers 0-6) |
| `results/jaffe_train_balanced/X_test.npy` | 43 test images (48x48) |
| `results/jaffe_train_balanced/y_test.npy` | 43 test labels |
| `results/jaffe_train_balanced/label_encoder.pkl` | Mapping between numbers and emotion names |

---

## What's Next? (Phase 2 Preview)

Now that our data is prepared, Phase 2 will:

1. **Extract Features (HOG)** - Instead of using raw pixels, we'll extract "features" that describe the important patterns in each face (edges, gradients, textures)

2. **Train a Classifier (SVM)** - We'll use a Support Vector Machine to learn the relationship between features and emotions

3. **Make Predictions** - Show the computer new faces and see if it can correctly identify the emotions

---

---

# Phase 2: Model Development

Now that our data is prepared, it's time to teach the computer! This phase has two main parts:
1. **Feature Extraction** - Converting images into meaningful descriptions
2. **Training a Classifier** - Teaching the computer to recognize patterns

---

## Why Not Just Use Raw Pixels?

You might wonder: "We already have images as numbers. Why can't we just feed those to the computer?"

We could, but it's not very effective. Here's why:

Imagine describing a happy face to a friend:
- **Bad description:** "Pixel 1 is 0.45, pixel 2 is 0.47, pixel 3 is 0.52..."
- **Good description:** "The corners of the mouth curve upward, the cheeks are raised..."

Raw pixels are like the bad description - lots of numbers with no clear meaning. **Features** are like the good description - they capture the *important patterns* that matter for recognition.

---

## Step 1: HOG Feature Extraction

### What is HOG?

**HOG** = Histogram of Oriented Gradients

Don't worry about the fancy name. Let's break it down:

### What is a Gradient?

A **gradient** is just a fancy word for "change". In an image, gradients show where brightness changes - which means **edges**.

```
Imagine a simple image:

Dark  | Light        The | line is an EDGE
██████|░░░░░░        (brightness changes suddenly)
██████|░░░░░░
██████|░░░░░░        Gradient points: Dark → Light (→)
```

Edges are super important because they show the **shape** of things - the outline of a nose, the curve of a smile, the furrow of an angry brow.

### What is an Orientation?

**Orientation** = direction. Gradients can point in different directions:

```
→  Horizontal edge (left to right)
↑  Vertical edge (bottom to top)
↗  Diagonal edge
↘  Another diagonal
```

### What is a Histogram?

A **histogram** is a way of counting things in bins.

Imagine sorting colored balls into buckets:
- Red bucket: 5 balls
- Blue bucket: 3 balls
- Green bucket: 7 balls

HOG does the same thing with gradient directions:
- 0° bucket: 12 gradients
- 20° bucket: 8 gradients
- 40° bucket: 15 gradients
- ... (9 direction buckets total)

### How HOG Works (Step by Step)

In [extractHOGFeatures.py](../src/extractHOGFeatures.py):

```python
from skimage.feature import hog

feature_vector = hog(
    image,
    orientations=9,         # 9 direction buckets
    pixels_per_cell=(8, 8), # Divide image into 8x8 pixel cells
    cells_per_block=(2, 2)  # Group cells into 2x2 blocks
)
```

**Step 1: Divide into Cells**

The 48x48 image is divided into small 8x8 pixel cells:
```
┌────┬────┬────┬────┬────┬────┐
│8x8 │8x8 │8x8 │8x8 │8x8 │8x8 │
├────┼────┼────┼────┼────┼────┤
│8x8 │8x8 │8x8 │8x8 │8x8 │8x8 │
├────┼────┼────┼────┼────┼────┤
│... │... │... │... │... │... │
└────┴────┴────┴────┴────┴────┘

48 ÷ 8 = 6 cells across
6 × 6 = 36 cells total
```

**Step 2: Calculate Gradients in Each Cell**

For each cell, HOG calculates:
- How strong are the edges?
- What direction do they point?

**Step 3: Create Histogram for Each Cell**

Count how many gradients point in each of 9 directions:
```
Cell histogram: [12, 8, 15, 3, 7, 22, 5, 9, 11]
                 ↑   ↑   ↑   ...
                 0°  20° 40° ... (9 directions)
```

**Step 4: Normalize Across Blocks**

Groups of 2x2 cells are normalized together. This makes HOG work even if lighting changes (a face in bright light vs. shadow).

**Step 5: Combine Everything**

All the histograms are joined into one long feature vector:

```
Input:  48x48 image (2,304 pixels)
Output: 900 HOG features

Each image is now described by 900 numbers that capture
the edge patterns - much more meaningful than raw pixels!
```

### Why HOG is Perfect for Faces

Faces have distinctive edge patterns:
- **Eyebrows**: Horizontal edges above the eyes
- **Eyes**: Strong edges around the eyelids
- **Nose**: Vertical edge down the middle
- **Mouth**: Horizontal edges, curves when smiling
- **Wrinkles**: Extra edges when angry or surprised

HOG captures all these patterns!

### Visual Example

```
Happy Face HOG:                  Angry Face HOG:
- Strong upward curves           - Strong downward curves
  at mouth corners                 at mouth corners
- Raised cheek edges             - Vertical furrow between
- Wide eye opening edges           eyebrows
                                 - Narrowed eye edges
```

---

## Step 2: Training the SVM Classifier

### What is a Classifier?

A **classifier** is an algorithm that learns to sort things into categories.

Think of it like training a dog:
- Show it many examples: "This is a ball" (tennis ball, basketball, soccer ball...)
- Eventually, it learns the pattern and can identify new balls it's never seen

### What is SVM?

**SVM** = Support Vector Machine

Despite the intimidating name, the concept is simple.

### The Basic Idea

Imagine you have two groups of points on a page - red dots and blue dots. You want to draw a line that separates them:

```
                    Good line!
    Red  Blue       ↓
    ●     ○        ●  |  ○
    ●     ○        ●  |  ○
    ●     ○        ●  |  ○
    ●     ○        ●  |  ○
```

SVM finds the **best possible line** - the one that:
1. Correctly separates the groups
2. Has the **maximum margin** (biggest gap between the line and nearest points)

### What are Support Vectors?

The points closest to the line are called **support vectors**. They "support" or define where the line goes:

```
         margin
         ←────→
    ●    ●̲ | ○̲    ○
    ●    ↑   ↑    ○
         Support vectors
         (closest points to the line)
```

### But Wait - What About Curves?

Real data isn't always separable by a straight line:

```
         ●  ○  ●
         ○  ●  ○
         ●  ○  ●

Can't draw a straight line to separate these!
```

This is where the **kernel trick** comes in.

### The RBF Kernel

**RBF** = Radial Basis Function

The kernel trick is like adding a dimension. Imagine the points above are on a flat table. Now push the table up in certain spots:

```
Side view:
         /\
        /  \        Now we CAN draw a
       ●    ●       separating plane!
      ○      ○
```

The RBF kernel does this mathematically - it transforms the data so that a "curved" boundary becomes possible.

### The Code

In [trainSVMClassifier.py](../src/trainSVMClassifier.py):

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',           # Use RBF kernel for curved boundaries
    class_weight='balanced', # Give equal importance to all emotions
    random_state=42          # For reproducibility
)

# Train: Show the model examples and correct answers
model.fit(X_train_hog, y_train)

# Predict: Ask the model to classify new examples
predictions = model.predict(X_test_hog)
```

### What Does `class_weight='balanced'` Do?

Even though we used SMOTE, some classes might still be slightly harder to classify. Setting `class_weight='balanced'` tells the SVM:

> "Treat mistakes on rare classes as MORE serious than mistakes on common classes"

It's like a teacher giving extra attention to struggling students.

### How SVM Learns Emotions

During training, SVM learns boundaries between emotions in the 900-dimensional HOG feature space:

```
Simplified 2D visualization:

    Happy ●        The SVM learns curved
          ●\       boundaries that separate
    Sad ○   \      each emotion region
        ○    \●
              ● Surprise
```

In reality, there are 900 dimensions (one for each HOG feature), and SVM finds complex boundaries in this space.

---

## Phase 2 Results

### Files Created

| File | Purpose |
|------|---------|
| `src/extractHOGFeatures.py` | Extract HOG features from images |
| `src/trainSVMClassifier.py` | Train and evaluate SVM classifier |

### Data Created

| File | Contents |
|------|----------|
| `results/jaffe_hog_features/X_train_hog.npy` | 175 training samples × 900 HOG features |
| `results/jaffe_hog_features/X_test_hog.npy` | 43 test samples × 900 HOG features |
| `results/jaffe_svm_model/svm_model.pkl` | Trained SVM model |

### Model Performance

```
Training Accuracy: 84.57%
Test Accuracy:     41.86%
```

Wait - why is training accuracy so much higher than test accuracy?

---

## Understanding Overfitting

### What is Overfitting?

**Overfitting** = The model memorized the training examples instead of learning general patterns.

It's like a student who memorizes answers to practice problems but can't solve new problems.

```
Training: "What's 2+2?" → "4" ✓ (memorized)
Training: "What's 3+3?" → "6" ✓ (memorized)
Test:     "What's 5+5?" → "?" ✗ (never seen, doesn't know)
```

### Why Did This Happen?

1. **Tiny dataset**: Only 175 training samples for 900 features
   - Rule of thumb: You need at least 10× more samples than features
   - We have: 175 samples for 900 features (way too few!)

2. **Small test set**: Only 43 images (about 6 per emotion)
   - Getting 1 wrong = ~17% drop in accuracy for that class

3. **Similar emotions**: Angry/Disgust look alike, Sad/Neutral look alike
   - Even humans confuse these!

### This is Normal!

For traditional ML on JAFFE, 40-60% test accuracy is common. The dataset is simply too small for robust learning. This is a known limitation you should mention in your report.

---

---

# Phase 3: Model Evaluation

After training the model, we need to evaluate how well it performs. This phase creates visualizations and metrics for your report.

---

## Evaluation Metrics Explained

### Accuracy

**Accuracy** = Percentage of correct predictions

```
Accuracy = Correct Predictions / Total Predictions

Example:
- 43 test images
- 18 correct predictions
- Accuracy = 18/43 = 41.86%
```

Simple but can be misleading. If 90% of your data is "Happy", a model that always guesses "Happy" gets 90% accuracy!

### Precision

**Precision** = "When the model says X, how often is it right?"

```
Precision = True Positives / (True Positives + False Positives)

Example for "Happy":
- Model predicted "Happy" 7 times
- 6 were actually Happy (True Positives)
- 1 was actually Neutral (False Positive)
- Precision = 6/7 = 86%

"When the model says Happy, it's right 86% of the time"
```

### Recall

**Recall** = "Of all actual X, how many did the model find?"

```
Recall = True Positives / (True Positives + False Negatives)

Example for "Happy":
- There are 6 Happy faces in test set
- Model found all 6 of them
- Recall = 6/6 = 100%

"The model found 100% of the Happy faces"
```

### F1 Score

**F1 Score** = Balance between Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

For "Happy":
F1 = 2 × (0.86 × 1.00) / (0.86 + 1.00) = 0.92

High F1 means both precision AND recall are good.
```

### The Precision-Recall Trade-off

Think of it like a security guard:
- **High Precision, Low Recall**: Very careful, only stops people they're SURE are intruders. Misses some actual intruders.
- **High Recall, Low Precision**: Stops everyone suspicious. Catches all intruders but also many innocent people.
- **Good F1**: Balanced - catches most intruders without too many false alarms.

---

## The Confusion Matrix

### What is a Confusion Matrix?

A **confusion matrix** shows exactly what the model predicted vs. what was correct.

```
                    PREDICTED
                 Angry  Happy  Sad
            ┌─────────────────────┐
     Angry  │   0      0      1   │  ← 0 correct, 1 confused with Sad
ACTUAL      ├─────────────────────┤
     Happy  │   0      6      0   │  ← 6 correct! (diagonal)
            ├─────────────────────┤
     Sad    │   0      0      1   │  ← 1 correct
            └─────────────────────┘
```

**Reading the matrix:**
- **Diagonal** (top-left to bottom-right): Correct predictions
- **Off-diagonal**: Mistakes (confusions)
- **Rows**: Actual emotion
- **Columns**: Predicted emotion

### Our Results

```
                    PREDICTED
            Angry Disgust Fear Happy Neutral Sad Surprise
        ┌───────────────────────────────────────────────┐
 Angry  │  0      4      0    0      1      1     0    │
Disgust │  3      2      1    0      0      0     0    │
 Fear   │  0      2      2    0      2      0     1    │
ACTUAL  │                                              │
 Happy  │  0      0      0    6      0      0     0    │
Neutral │  1      1      0    1      3      0     0    │
  Sad   │  0      2      1    0      2      1     0    │
Surprise│  0      0      0    0      2      0     4    │
        └───────────────────────────────────────────────┘
```

**Key Insights:**
- **Happy** is perfectly classified (6/6 on diagonal) - smiles are distinctive!
- **Angry** is never correctly classified (0 on diagonal) - always confused with Disgust
- **Angry/Disgust** are confused with each other (4 Angry→Disgust, 3 Disgust→Angry)
- **Surprise** is fairly good (4/6 correct)

### Why Are Some Emotions Confused?

Looking at the faces, it makes sense:

```
Angry:    Furrowed brow, tight lips
Disgust:  Furrowed brow, raised upper lip

These look similar! Both have:
- Lowered/furrowed eyebrows
- Tension in the face
```

Even humans sometimes confuse these emotions.

---

## Visualizations Created

In [evaluateJaffeModel.py](../src/evaluateJaffeModel.py), we create:

### 1. Confusion Matrix Heatmap

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

A color-coded version of the confusion matrix:
- Darker blue = more predictions in that cell
- Makes it easy to spot patterns visually

### 2. Training vs Test Accuracy

```python
plt.bar(['Training', 'Testing'], [84.6, 41.9])
```

Bar chart showing the overfitting gap:
- Green bar: Training accuracy (84.6%)
- Blue bar: Test accuracy (41.9%)

The big gap shows overfitting.

### 3. Per-Class Accuracy

```python
# Accuracy for each emotion separately
```

Bar chart showing which emotions are easiest/hardest:
- Happy: 100% (easiest)
- Surprise: 67%
- Neutral: 50%
- Disgust: 33%
- Fear: 29%
- Sad: 17%
- Angry: 0% (hardest)

Color-coded: Green = good, Red = bad

### 4. Sample Predictions

```python
# Show 6 test images with actual vs predicted labels
```

Visual examples of the model's predictions:
- Green title = Correct prediction
- Red title = Wrong prediction

This is required for your report - shows the model "in action".

---

## Phase 3 Results

### Files Created

| File | Purpose |
|------|---------|
| `src/evaluateJaffeModel.py` | Generate all evaluation visualizations |

### Visualizations Created

| File | Description |
|------|-------------|
| `results/jaffe_evaluation/confusion_matrix.png` | Heatmap of predictions vs actual |
| `results/jaffe_evaluation/accuracy_comparison.png` | Training vs Test accuracy bars |
| `results/jaffe_evaluation/per_class_accuracy.png` | Accuracy per emotion |
| `results/jaffe_evaluation/sample_predictions.png` | 6 example predictions with images |
| `results/jaffe_evaluation/evaluation_report.txt` | Full text report |

---

---

# Complete Project Summary

## The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                  │
│                    DATA PREPARATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Images        Preprocess         Split           Balance   │
│   (JAFFE)     →    (Gray,48x48)   →   (80/20)    →    (SMOTE)   │
│   213 images       Normalized         170/43          175/43     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 2                                  │
│                    MODEL DEVELOPMENT                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Images (48x48)      HOG Features        SVM Training           │
│   175 training   →    900 features   →    RBF Kernel             │
│   43 test             per image           Trained Model          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 3                                  │
│                    MODEL EVALUATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Predictions         Metrics            Visualizations          │
│   on test set    →    Accuracy      →    Confusion Matrix        │
│                       Precision          Sample Predictions      │
│                       Recall             Accuracy Charts         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## All Files Created

### Source Code (`src/`)

| File | Phase | Purpose |
|------|-------|---------|
| `loadJaffeDataset.py` | 1 | Load images and labels |
| `jaffeInfo.py` | 1 | Explore and visualize dataset |
| `checkJaffeClassImbalance.py` | 1 | Check class balance |
| `TrainJaffeSets.py` | 1 | Preprocess and split data |
| `handleJaffeImbalance.py` | 1 | Apply SMOTE balancing |
| `extractHOGFeatures.py` | 2 | Extract HOG features |
| `trainSVMClassifier.py` | 2 | Train SVM classifier |
| `evaluateJaffeModel.py` | 3 | Generate evaluation visualizations |

### Results (`results/`)

| Directory | Contents |
|-----------|----------|
| `jaffe_info/` | Dataset distribution and sample images |
| `jaffe_train/` | Original preprocessed train/test split |
| `jaffe_train_balanced/` | SMOTE-balanced training data |
| `jaffe_hog_features/` | HOG feature vectors |
| `jaffe_svm_model/` | Trained SVM model |
| `jaffe_evaluation/` | All visualizations and reports |

## Final Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 84.57% |
| Test Accuracy | 41.86% |
| Best Emotion (Happy) | 100% |
| Worst Emotion (Angry) | 0% |

## Limitations to Mention in Report

1. **Small Dataset**: JAFFE has only 213 images - too few for robust learning
2. **Overfitting**: Large gap between training and test accuracy
3. **Similar Emotions**: Angry/Disgust and Sad/Neutral are often confused
4. **Single Demographics**: JAFFE only has Japanese female subjects

## Potential Improvements

1. **More Data**: Use larger datasets like CK+ or FER2013
2. **Cross-Validation**: Instead of single train/test split
3. **Hyperparameter Tuning**: GridSearchCV for SVM parameters
4. **Different Features**: Try LBP or Gabor filters
5. **Ensemble Methods**: Combine multiple classifiers

---

## Glossary

| Term | Simple Definition |
|------|-------------------|
| **Class** | A category (Happy, Sad, Angry, etc.) |
| **Class Imbalance** | Having way more examples of some classes than others |
| **Classifier** | Algorithm that sorts things into categories |
| **Confusion Matrix** | Table showing predictions vs actual labels |
| **Dataset** | A collection of examples used for teaching |
| **F1 Score** | Balance between precision and recall |
| **Feature** | A measurable property of the data (pixel value, edge direction, etc.) |
| **Gradient** | Change in brightness; shows edges |
| **Grayscale** | Black and white image (no colors) |
| **Histogram** | Counting things into bins/buckets |
| **HOG** | Histogram of Oriented Gradients - feature extraction method |
| **Kernel** | Mathematical trick to handle curved boundaries |
| **Label** | The correct answer (the emotion name) |
| **Label Encoding** | Converting text labels to numbers |
| **Normalization** | Scaling values to a standard range (like 0 to 1) |
| **Overfitting** | Model memorizes training data instead of learning patterns |
| **Pixel** | A tiny square in an image; the smallest unit |
| **Precision** | "When model predicts X, how often is it correct?" |
| **Preprocessing** | Cleaning and preparing data before learning |
| **RBF** | Radial Basis Function - a type of kernel for SVM |
| **Recall** | "Of all actual X, how many did the model find?" |
| **SMOTE** | Technique to create synthetic samples for rare classes |
| **Stratified Split** | Keeping the same proportion of classes in train and test sets |
| **Support Vector** | Points closest to the decision boundary |
| **SVM** | Support Vector Machine - a classification algorithm |
| **Test Set** | Data used ONLY to evaluate (never for training) |
| **Training Set** | Data used to teach the computer |
| **X** | Input data (the images) |
| **y** | Output labels (the correct emotions) |

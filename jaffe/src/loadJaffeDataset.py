import os #uhh file path management
import cv2 #image reading

jaffeRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #get the root dir for the project
jaffeDatasetDir = os.path.join(jaffeRootDir, 'data', 'jaffe dataset', 'jaffe', 'jaffe') #get the path to the dataset

jaffeEmotionMap = {
    'HA': 'Happy',
    'SA': 'Sad',
    'SU': 'Surprise',
    'AN': 'Angry',
    'DI': 'Disgust',
    'FE': 'Fear',
    'NE': 'Neutral'
}

def loadJaffeImages(jaffeDatasetDir):
    images = []
    labels = []

    #get all image files
    imageFiles = [f for f in os.listdir(jaffeDatasetDir) if f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg')]

    print(f"Found {len(imageFiles)} images in JAFFE dataset")

    for fileName in imageFiles:
        parts = fileName.split('.')
        if len(parts) >= 2:
            emotionCode = parts[1][:2] #get the emotion code from filename

            if emotionCode in jaffeEmotionMap:
                imgPath = os.path.join(jaffeDatasetDir, fileName)
                image = cv2.imread(imgPath)

                if image is not None:
                    images.append(image)
                    labels.append(jaffeEmotionMap[emotionCode])

    print(f"Successfully loaded {len(images)} images from JAFFE dataset")
    return images, labels



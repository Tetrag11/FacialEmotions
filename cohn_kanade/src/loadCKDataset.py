import os #uhh file path management
import cv2 #image reading

ckRootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #get the root dir for the project
ckDatasetDir = os.path.join(ckRootDir, 'data', 'Cohn Kanade') #get the path to the dataset

ckEmotionMap ={
    'anger': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy',
    'sadness': 'Sad',     
    'surprise': 'Surprise'
}

def loadCkImages(ckDatasetDir):
    images = []
    labels = []
    
    for folderName, emotionLabel in ckEmotionMap.items():
        folderPath = os.path.join(ckDatasetDir, folderName) #loop through the emotion folders
        
        if not os.path.exists(folderPath):
            continue
        
        #get the imagefiles
        imageFiles = [f for f in os.listdir(folderPath) if f.endswith('.png')] #get all the files inside the listed folder if they ends with .png
        
        
        for fileName in imageFiles:
            imgPath = os.path.join(folderPath, fileName)
            image = cv2.imread(imgPath)
            
            if image is not None:
                images.append(image)
                labels.append(emotionLabel)
    
    print(f"Successfully loaded {len(images)} images from Cohn-Kanade dataset")            
    return images, labels        
        



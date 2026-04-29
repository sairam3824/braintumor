import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(dataset_path, augment=False):

    categories = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

    data = []
    labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for label, category in enumerate(categories):
        path = os.path.join(dataset_path, category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                continue
                
            # 1️⃣ Apply CLAHE for contrast standardization
            image = clahe.apply(image)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            
            # 2️⃣ Normalize
            img_norm = image / 255.0

            data.append(img_norm)
            labels.append(label)
            
            # 3️⃣ Data Augmentation
            if augment:
                flipped = cv2.flip(image, 1)
                data.append(flipped / 255.0)
                labels.append(label)

    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    return data, labels
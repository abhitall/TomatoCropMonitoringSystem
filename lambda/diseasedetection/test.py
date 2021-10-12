import os
import numpy as np
from base64 import b64decode, encodestring
from traceback import print_exc
from tensorflow.io import decode_image
from tensorflow.image import resize
from tensorflow.keras.models import load_model

model = load_model('model.h5')
classes = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict_class(b64_image):
    img = decode_image(b64decode(b64_image), channels=3)
    img = resize(img, size=(256, 256))
    img = np.expand_dims(img, axis=0)
    result = classes[np.argmax(model.predict(img))]
    return result

fileslist = [[os.path.join(r, file) for file in f if file.endswith(".jpg") or file.endswith(".JPG")] for r, d, f in os.walk("C:/Users/tallu/Desktop/TomatoCropDiseaseDetection/TomatoData1")]

for file_path in fileslist:
    image = open(file_path, 'rb')
    image_read = image.read()
    image_64_encode = encodestring(image_read)
    print(predict_class(image_64_encode))
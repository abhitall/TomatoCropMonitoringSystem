import numpy as np
from json import loads, dumps
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageOps
from traceback import print_exc
from tensorflow.keras.models import load_model

model = load_model('model.h5')
classes = ['Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite']

def predict_class(b64_image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(BytesIO(b64decode(b64_image)))
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    normalized_img = (np.asarray(img).astype(np.float32) / 127.0) - 1
    data[0] = normalized_img
    result = classes[np.argmax(model.predict(data))]
    return result

def handler(event, context):
    try:
        body = loads(event['body'])
        b64_image = body['img']
        b64_image = b64_image[b64_image.find(",")+1:] + "==="
        return {
            "statusCode": 200,
        	"headers":{
        		"Access-Control-Allow-Origin":"*",
        	},
            "body": dumps({
                "predictedClass": predict_class(b64_image)
            })
        }
    except Exception:
        return {
            "statusCode": 500,
        	"headers":{
        		"Access-Control-Allow-Origin":"*",
        	},
            "body": dumps({
                "error": print_exc()
            })
        }

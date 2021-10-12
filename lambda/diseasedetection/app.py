import numpy as np
from json import loads, dumps
from base64 import b64decode
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

def handler(event, context):
    try:
        body = loads(event['body'])
        b64_image = body['img']
        # b64_image = b64_image[b64_image.find(",")+1:] + "==="
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

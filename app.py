#imports
import os
import numpy as np
# tf Keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.python.keras.backend import dtype
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model=None

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x.astype("float") / 255.0 #normailizing the image between [0-255]
    x = np.expand_dims(x,0) #adding batch size of 1 to get shape (1,img_height,img_width,channels=3)
    # print(x.shape)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        os.remove(file_path)
        if preds[0][0]>=0.5:
            return "Recyclable waste"
        else:
            return "Organic Waste"
    return None


if __name__ == '__main__':
    model=load_model('model.h5')
    app.run(host="0.0.0.0", port=5000, debug=True)

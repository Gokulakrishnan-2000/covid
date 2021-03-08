from flask import Flask, render_template, request, url_for

from werkzeug.utils import secure_filename
import os

import numpy as np
import tensorflow as tf 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import PIL

app = Flask(__name__)

model_path = "./model/20210307-17041615136671-Covid_pred_model.h5"
model = tf.keras.models.load_model(model_path)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/upload-image',methods = ['POST'])
def upload_image():
    UPLOAD_FOLDER = './static/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    filename=None
    error_msg=None
    file = request.files['image']
    if file.filename == '':
        error_msg="Please Upload Any Image"
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_path="./static/uploads/{}".format(filename)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        New_pred = np.argmax(classes, axis=1)
        if New_pred==[1]:
            print('Prediction: Normal')
            label = "Normal"
        else:
            print('Prediction: Corona')
            label = "Corona"

        return render_template("result.html",label=label,img_path=img_path)



    
if __name__ == "__main__":
    app.run(debug=True)

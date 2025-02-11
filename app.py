# from flask import Flask, redirect, url_for, request, render_template, session
# import sys
# import os
# import re
# import numpy as np
# from werkzeug.utils import secure_filename

# # Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import load_img
# #from tensorflow.keras.utils import load_img
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2
# #from PIL.Image import load_img

# # load model
# model_path = 'wound_vgg19five.h5'
# model = load_model(model_path)

# app = Flask(__name__)

# # Define the directory where uploaded files will be stored
# app.config['UPLOAD_FOLDER'] = os.path.join('upload/')
# app.config['UPLOAD'] = os.path.join('')
# # Allow file uploads of up to 16 megabytes
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# # Ensure that file extensions are limited to images only
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

 
# @app.route("/")
# def main():
#     return render_template('main.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     return render_template('login.html')

# @app.route("/logout")
# def logout():
#     return render_template('main.html')


# @app.route('/performance')
# def performance():
#     return render_template('performance.html')


# @app.route("/index")
# def index():
#     return render_template('index.html')

# #creates function

# def predictions(img_path, model):
#     img = load_img(img_path, target_size=(224,224,3))

#     x = image.img_to_array(img)
#     x = x/255
#     x = np.expand_dims(x, axis=0)
#     pred = np.argmax(model.predict(x)[0], axis=-1)
#     print(pred)

#     if pred == 0:
#         preds = 'Accident_images'
    
#     elif pred == 1:
#         preds = "Burns"
#     elif pred == 2:
#         preds = " Cut"
#     elif pred == 3:
#         preds = "Diabetic_foot_ulcers" 
#     elif pred ==4:
#         preds = "Ingrown_nails"
    

#     return preds
#     print(preds)
    


# @app.route("/predicted", methods=['POST'])
# def predicted():
#     # Get the uploaded image file from the form
#     uploaded_file = request.files['imagefile']
#     # print(uploaded_file)
    
  
    
#     # Save the file to a temporary directory
#     filename = secure_filename(uploaded_file.filename)
#     print(filename)
#     uploaded_file.save(filename)
#     # print(filename)
#     img = cv2.imread(filename)
#     path = 'C:/Users/rizwan/Documents/major project/ip05/static/'+filename
#     cv2.imwrite(path, img)
#     img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     # print(img_path)
    
#     #uploaded_file.save(img_path)
    
#     img_upload = cv2.imread(filename)
#     # print(img_upload)

# # Add a third dimension to the image array to indicate that it is a grayscale image
#     img_upload = np.expand_dims(img_upload, axis=2)
#     rgb_image = np.repeat(img_upload, 3, axis=2)
    
 
#     # Save the image to the 'static' directory
#     img = os.path.join('static/', filename)
#     #mpimg.imsave(img, rgb_image)
#     print(img)
#     # Save the grayscale image to the 'static' directory
    

#     # Get the prediction for the uploaded image
#     prediction = predictions(img_path, model)
    
#     # Pass the image file path and prediction result to the template
#     return render_template('result.html', prediction=prediction, img=img)


# if __name__ == '__main__':
#     app.run()


from flask import Flask, redirect, url_for, request, render_template, session
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load model
model_path = 'wound_vgg19five.h5'
model = load_model(model_path)

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template('main.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('main.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route("/index")
def index():
    return render_template('index.html')

def predictions(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    pred = np.argmax(model.predict(x)[0], axis=-1)

    if pred == 0:
        preds = 'Accident_images'
    elif pred == 1:
        preds = "Burns"
    elif pred == 2:
        preds = "Cut"
    elif pred == 3:
        preds = "Diabetic_foot_ulcers"
    elif pred == 4:
        preds = "Ingrown_nails"
    
    return preds

@app.route("/predicted", methods=['POST'])
def predicted():
    if 'imagefile' not in request.files:
        return "No file part"
    
    uploaded_file = request.files['imagefile']
    if uploaded_file.filename == '':
        return "No selected file"

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(img_path)
        
        # Get the prediction for the uploaded image
        prediction = predictions(img_path, model)
        
        # Pass the image file path and prediction result to the template
        img = os.path.join('static/', filename)
        cv2.imwrite(img, cv2.imread(img_path))
        
        return render_template('result.html', prediction=prediction, img=img)
    
    return "Invalid file"

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()

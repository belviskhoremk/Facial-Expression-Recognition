from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('FER.h5')


# Define the route for the prediction endpoint
@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Read the file from the request
        file = request.files['file'].read()

        # Convert the file to an image
        img = Image.open(io.BytesIO(file))
        img = img.convert('L')

        # Resize the image to the input size expected by the model
        img = img.resize((48, 48))

        # Convert the image to a numpy array
        x = np.array(img)

        # Convert grayscale image to 3-channel (RGB)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        # Resize the image to the input size expected by the model
        x = cv2.resize(x, (48, 48))

        # Normalize the pixel values
        x = x / 255.0

        # Add a batch dimension to the array
        x = np.expand_dims(x, axis=0)

        # Make the prediction
        pred = model.predict(x)

        # Get the predicted class label
        label = np.argmax(pred[0])
        result = label
        result=int(result)
        if result == 0:
            output = 'angry'
        elif result==1:
            output='disgust'
        elif result==2:
            output='fear'
        elif result==3:
            output='happy'
        elif result==4:
            output='neutral'
        elif result==5:
            output='sad'
        elif result==6:
            output='surprise'


        # Return the result as a JSON response
        return jsonify({'result': output})

    return render_template('deploy.html')





if __name__ == '__main__':
    app.run(debug=True)

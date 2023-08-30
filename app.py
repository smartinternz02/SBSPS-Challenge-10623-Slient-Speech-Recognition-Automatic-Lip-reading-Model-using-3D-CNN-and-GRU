from flask import Flask, render_template, request
import os
import imageio
import tensorflow as tf
from utils import load_data,num_to_char
from modelutil import load_model

app = Flask(__name__)

model = load_model()

@app.route('/', methods=['GET'])
def start():
    return render_template('ik.html')

@app.route('/new', methods=['GET'])
def new_page():
    return render_template('new.html')
    
@app.route('/new2', methods=['GET'])
def new_page2():
    return render_template('new2.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    animation_url = None
    decoded_text = None

    if request.method == 'POST':
        uploaded_file = request.files['video']
        if uploaded_file:
            uploaded_filename = uploaded_file.filename  # Extract the uploaded file's name
            video_path = os.path.join('static', uploaded_filename)
            uploaded_file.save(video_path)

            video, annotations = load_data(tf.convert_to_tensor(video_path))
           

            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

           
            decoded_text = converted_prediction

    # Return the result outside the if block
    return render_template('index.html', decoded_text=decoded_text)



if __name__ == '__main__':
    app.run(debug=True)

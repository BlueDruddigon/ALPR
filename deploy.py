import os

import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from detect import pipeline

UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED = os.path.join('static', 'processed')

app = Flask(__name__)
app.config['UPLOAD'] = UPLOAD_FOLDER
app.config['PROCESSED'] = PROCESSED
app.secret_key = 'APP'


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        file = request.files['uploaded-file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)
        img = os.path.join(app.config['PROCESSED'], filename)
        raw_image = cv2.imread(img_path)
        height, width, _ = raw_image.shape
        processed, lpr, msg = pipeline(raw_image)
        cv2.imwrite(img, processed)
        return render_template('index.html', img=img, height=height, width=width, lpr=lpr, msg=msg)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

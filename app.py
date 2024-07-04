from fileinput import filename
from flask import Flask, request, redirect, flash, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
from flask_cors import CORS
from pathlib import Path

import os
import sys

from train_transfer_image_style import train_transfer_image_style

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

print(ROOT_DIR, file=sys.stderr)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # Max file 512MB

cors = CORS(app, resources={r"*": {"origins": "*"}})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads', methods=['GET'])
def uploads():
    # Verify the output folder exists and create one if it doesn't
    Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    files = os.listdir(app.config['OUTPUT_FOLDER'])
    return render_template('uploads.html', files=files)

@app.route('/download_file/<name>')
def download_file(name):
    return send_from_directory(app.config["OUTPUT_FOLDER"], name)

@app.route('/submit/<int:epochs>/<int:width>/<int:height>', methods=['GET'])
def submit(epochs, width, height):
    return f'The inputs you provided are: {epochs}, {width} and {height}'

@app.route('/transfer_style/', methods=['POST'])
def transfer_style():
    epochs = request.form['epochSlider']
    width = int(request.form['widthSlider'])
    height = int(request.form['heightSlider'])

    print("Epochs: " + epochs, file=sys.stderr)
    print(f"{width} x {height}", file=sys.stderr)

    if request.method == 'POST':
        # check if the post request has the required images
        if 'source' not in request.files:
            flash('No source image!')
            # TODO: Throw an error to the user
            return redirect(request.url)
        if 'target' not in request.files:
            flash('No target image!')
            return redirect(request.url)

        source = request.files['source']
        target = request.files['target']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if source.filename == '':
            flash('No selected source image')
            return redirect(request.url)
        if target.filename == '':
            flash('No selected target image')
            return redirect(request.url)
        
        # Create the upload folder if it does not exist
        Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

        if source and target and allowed_file(source.filename) and allowed_file(target.filename):
            source_filename = secure_filename(source.filename)
            source.save(os.path.join(app.config['UPLOAD_FOLDER'], source_filename))
            target_filename = secure_filename(target.filename)
            target.save(os.path.join(app.config['UPLOAD_FOLDER'], target_filename))

            # Verify the output folder exists and create one if it doesn't
            Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

            # Actually generate the style transfer image
            train_transfer_image_style(
                source,
                target,
                os.path.join(app.config['OUTPUT_FOLDER'],
                         f'{source.filename}-{target.filename}.png'),
                epochs,
                'vgg19',
                width,
                height)
        return redirect(url_for('download_file'))
    return redirect(url_for("uploads"))


if __name__ == "__main__":
    app.run()

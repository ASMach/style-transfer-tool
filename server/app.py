from fileinput import filename
from flask import Flask, request, redirect, flash, url_for, render_template, send_from_directory, make_response
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from pathlib import Path

import os
import sys
import json

from train_transfer_image_style import train_transfer_image_style
from crossdomain import crossdomain

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

@app.after_request
def build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads')
@cross_origin()
def uploads():
    # Verify the output folder exists and create one if it doesn't
    Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    files = os.listdir(app.config['OUTPUT_FOLDER'])
    return json.dumps(files)

@app.route('/download_file/<name>')
@cross_origin()
def download_file(name):
    return send_from_directory(app.config["OUTPUT_FOLDER"], name)

@app.route('/transfer_style/', methods=['POST'])
@crossdomain(origin='http://localhost:3000')
def transfer_style():
    epochs = int(request.form['epochSlider'])
    width = int(request.form['widthSlider'])
    height = int(request.form['heightSlider'])

    print(f"Epochs: {epochs}", file=sys.stderr)
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

            outfile = f'{source_filename}-{target_filename}.png'
            # Actually generate the style transfer image
            train_transfer_image_style(
                source,
                target,
                os.path.join(app.config['OUTPUT_FOLDER'],
                         outfile),
                epochs,
                'vgg19',
                width,
                height)
        return redirect(url_for('download_file', name=outfile))
    return redirect("/uploads")


if __name__ == "__main__":
    app.run()

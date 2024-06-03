from fileinput import filename
from flask import Flask, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
from flask_cors import CORS

import os

from train_transfer_image_style import train_transfer_image_style

UPLOAD_FOLDER = '/uploads'
OUTPUT_FOLDER = '/output'
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
    return "Hello, World from Flask!"


@app.route('/transfer_style/<int:epochs>/<int:width>/<int:height>', methods=['POST'])
def transfer_style(source=None, target=None, epochs=100, width=512, height=512):
    epochs = request.args.get('epochs', epochs)
    width = request.args.get('width', width)
    height = request.args.get('height', height)

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

        if source and target and allowed_file(source.filename) and allowed_file(target.filename):
            f = secure_filename(source)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], source))
            f = secure_filename(target)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], target))
            return redirect(url_for('download_file'))

        train_transfer_image_style(
            source,
            target,
            os.path.join(app.config['OUTPUT_FOLDER'],
                         f'{source}-{target}.png'),
            epochs,
            'vgg19',
            width,
            height)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run()

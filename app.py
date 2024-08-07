from celery_setup import make_celery
from celery.result import AsyncResult
from flask import Flask, request, redirect, flash, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_bootstrap import Bootstrap4
from pathlib import Path
from PIL import Image
from werkzeug.utils import secure_filename

import os
import redis
import sys
import time
import tqdm

from train_transfer_image_style import train_transfer_image_style

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

print(ROOT_DIR, file=sys.stderr)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

bootstrap = Bootstrap4(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # Max file 512MB
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_URL')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_URL')

cors = CORS(app, resources={r"*": {"origins": "*"}})

celery = make_celery(app)

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@celery.task(bind=True)
def long_task(self, total):
    task_id = self.request.id
    progress_bar = tqdm(total=total)
    
    for i in range(total):
        time.sleep(1)  # Simulate a task taking time
        progress_bar.update(1)
        redis_client.set(task_id, i + 1)
    
    progress_bar.close()
    return {'current': total, 'total': total, 'status': 'Task completed!'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads')
def uploads():
    # Verify the output folder exists and create one if it doesn't
    Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    files = os.listdir(app.config['OUTPUT_FOLDER'])
    return render_template('uploads.html', files=files)

@app.route('/status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        progress = redis_client.get(task_id)
        if progress:
            current = int(progress)
        else:
            current = 0
        response = {
            'state': task.state,
            'current': current,
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/thumb/<name>')
def thumb(name):
    return send_from_directory(app.config["OUTPUT_FOLDER"], os.path.join(name, 'thumb.png'))

@app.route('/image/<name>')
def image(name):
    return send_from_directory(app.config["OUTPUT_FOLDER"], os.path.join(name, 'image.png'))

@app.route('/transfer_style/', methods=['POST'])
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

            # Split file extensions from names
            source_filename = os.path.splitext(source_filename)[0]
            target_filename = os.path.splitext(target_filename)[0]

            # Create a folder for the source output image
            outpath = os.path.join(app.config['OUTPUT_FOLDER'], f'{source_filename}-{target_filename}')
            Path(outpath).mkdir(parents=True, exist_ok=True)

            outfile = 'image.png'
            # Actually generate the style transfer image
            task = train_transfer_image_style(
                source,
                target,
                os.path.join(outpath,
                         outfile),
                epochs,
                'vgg19',
                width,
                height)
            # Generate thumbnail
            full_outfile = os.path.join(outpath, outfile)
            im = Image.open(full_outfile)

            thumb_size = 128, 128
            
            im.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            im.save(os.path.join(outpath, 'thumb.png'), "PNG")
    return redirect("/uploads")


if __name__ == "__main__":
    app.run()

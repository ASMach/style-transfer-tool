# style-transfer-tool

USAGE

- Use the appropriate setup script (.sh for \*NIX systems including macOS, .bat for Windows)

BROWSER INTERFACE

- flask run
- open http://127.0.0.1:5000 in your browser. Use the displayed webpage to set up a style transfer.

COMMAND LINE MODE

- run main.py --source {STYLE_IMAGE_PATH} --target {IMAGE_TO_ALTER_STYLE} --outfile {OUTPUT_IMAGE} --epochs {TRAINING_ITERATION_COUNT}

Note that --epochs is an optional parameter, the default is 250.

The example file was generated with the following command:

python3 main.py --source ./transfer_source/candy.jpg --target ./test_img/elephant.jpg --outfile gen.png

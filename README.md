# style-transfer-tool

USAGE

- Use the appropriate setup script (.sh for \*NIX systems including macOS, .bat for Windows)
- run main.py --source {STYLE_IMAGE_PATH} --target {IMAGE_TO_ALTER_STYLE} --outfile {OUTPUT_IMAGE} --epochs {TRAINING_ITERATION_COUNT}

Note that --epochs is an optional parameter, the default is 250.

The example file was generated with the following command:

python3 main.py --source ./transfer_source/candy.jpg --target ./test_img/elephant.jpg --outfile gen.png

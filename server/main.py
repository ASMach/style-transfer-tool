import argparse

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import torch.optim as optim

import utils as utils

from PIL import Image
from torchvision.utils import save_image

from train_transfer_image_style import train_transfer_image_style

from tqdm import tqdm

from vgg_nets import VGG, VGG16, VGG19

content_urls = dict(
    oxford="./test_img/oxford.jpg",
    eiffel_tower="./test_img/eiffel_tower.jpg",
    elephant="./test_img/elephant.jpg",
    hotdog="./test_img/hotdog.jpg",
    office="./test_img/office.jpg",
    room="./test_img/room.jpg",
    traffic="./test_img/traffic.jpg",
    zebra="./test_img/zebra.jpg",
    bicycle="./test_img/imagenet_images/bicycle.jpg",
    dog="./test_img/imagenet_images/dog.jpg",
    Lion="./test_img/imagenet_images/Lion.jpg",
    ship="./test_img/imagenet_images/ship.jpg",
    tiger="./test_img/imagenet_images/tiger.jpg",
    yacht="./test_img/imagenet_images/yacht.jpg",
)
style_urls = dict(
    bouguereau_abduction_psyche="./transfer_source/bouguereau_abduction_psyche.jpeg",
    bouguereau_bacchante="./transfer_source/bouguereau_bacchante.jpeg",
    bouguereau_dawn="./transfer_source/bouguereau_dawn.jpeg",
    bouguereau_gabrielle_cot="./transfer_source/bouguereau_gabrielle_cot.jpeg",
    bouguereau_nymphs_and_satyr="./transfer_source/bouguereau_nymphs_and_satyr.jpeg",
    bouguereau_venus_anodyne="./transfer_source/bouguereau_venus_anodyne.jpeg",
    bouguereau_evening_mood="./transfer_source/bouguereau_evening_mood.jpeg",
    galaxy_of_musicians="./transfer_source/galaxy_of_musicians.jpeg",
    gerome_femme_circassienne_voilee="./transfer_source/gerome_femme_circassienne_voilee.jpeg",
    gerome_gladiator_death="./transfer_source/gerome_gladiator_death.jpeg",
    gerome_phryne="./transfer_source/gerome_phryne.jpeg",
    monalisa="./transfer_source/monalisa.jpeg",
    raphael_young_man="./transfer_source/raphael_young_man.jpeg",
    starry_night="./transfer_source/starry_night.jpg",
    feathers="./transfer_source/feathers.jpg",
    candy="./transfer_source/candy.jpg",
    the_scream="./transfer_source/the_scream.jpg",
    mosaic="./transfer_source/mosaic.jpg",
    la_muse="./transfer_source/la_muse.jpg",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="style source image name")
    parser.add_argument("--target", type=str, help="image to change style of")
    parser.add_argument("--outfile", type=str,
                        help="image output file path and name")
    parser.add_argument("--epochs", type=int,
                        help="number of style transfer cycles", default=250)
    parser.add_argument("--model", type=str,
                        choices=['vgg16', 'vgg19', 'vgg'], default='vgg')
    parser.add_argument("--width", type=int,
                        help="width of image for processing", default=512)
    parser.add_argument("--height", type=int,
                        help="height of image for processing", default=512)

    args = parser.parse_args()

    print(f"Model Type: {args.model}")

    train_transfer_image_style(
        args.source, args.target, args.outfile, args.epochs, args.model, args.width, args.height)

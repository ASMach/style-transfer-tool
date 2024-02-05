import argparse

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import torch.optim as optim

import utils as utils

from PIL import Image
from torchvision.utils import save_image

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

device = utils.check_device()

# initialize the paramerters required for fitting the model
lr = 0.004
alpha = 8
beta = 70

# Core of the transfer system


def train_transfer_image_style(source, target, outfile, epoch=250, model_name='vgg'):
    # Loading the original and the style image
    content_image = utils.image_loader(target, device)
    style_image = utils.image_loader(source, device)

    utils.show_images([source, target],
                      titles=['Style image', 'Content image'])

    # Creating the generated image from the original image
    generated_image = content_image.clone().requires_grad_(True)

    # Load the model to the GPU
    model = None
    if model_name is 'vgg16':
        model = VGG16().to(device).eval()
    elif model_name is 'vgg19':
        model = VGG19().to(device).eval()
    elif model_name is 'vgg':
        model = VGG().to(device).eval()

    # using adam optimizer and it will update the generated image not the model parameter
    optimizer = optim.Adam([generated_image], lr=lr)

    # iterating for number of training epochs
    for e in tqdm(range(epoch)):
        print(f'epoch {e}')
        # extracting the features of generated, content and the original required for calculating the loss
        gen_features = model(generated_image)
        orig_feautes = model(content_image)
        style_featues = model(style_image)

        # iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
        total_loss = utils.calculate_loss(
            gen_features, orig_feautes, style_featues, alpha, beta)
        # optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print the image and save it after each 100 epoch
        if (e/100):
            save_image(generated_image, outfile)

    plt.imshow(mpimg.imread(outfile))


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

    args = parser.parse_args()

    train_transfer_image_style(
        args.source, args.target, args.outfile, args.epochs, args.model)

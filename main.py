import argparse

import torch
import platform

import torchvision.transforms as transforms
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np

from PIL import Image
from torchvision.utils import save_image

from tqdm import tqdm

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

# initialize the paramerters required for fitting the model
lr = 0.004
alpha = 8
beta = 70

# Helper functions


def show_images(images, titles=('',), dim=1024):
    n = len(images)
    for i in range(n):
        img = mpimg.imread(images[i])
        plt.imshow(img, aspect='equal')
        plt.title(titles[i] if len(titles) > i else '')
        plt.show()


def image_loader(path):
    image = Image.open(path)
    # defining the image transformation steps to be performed before feeding them to the model
    loader = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()])
    # The preprocessing steps involves resizing the image and then converting it to a tensor
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
    content_l = torch.mean((gen_feat-orig_feat)**2)
    return content_l


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    batch_size, channel, height, width = gen.shape

    G = torch.mm(gen.view(channel, height*width),
                 gen.view(channel, height*width).t())
    A = torch.mm(style.view(channel, height*width),
                 style.view(channel, height*width).t())

    # Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l = torch.mean((G-A)**2)
    return style_l


def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss = content_loss = 0
    for gen, cont, style in zip(gen_features, orig_feautes, style_featues):
        # extracting the dimensions from the generated image
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)

    # calculating the total loss of e th epoch
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss


def check_device():
    if platform.system() == 'Darwin':
        print(f"Torch MPS Available: {torch.backends.mps.is_available()}")
        print(f"Torch MPS Built: {torch.backends.mps.is_built()}")
    else:
        print(torch.cuda.is_available())
        print(f"CUDA Devides: {torch.cuda.device_count()}")
        print(f"Current CUDA Index: {torch.cuda.current_device()}")

    if platform.system() == 'Darwin':
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if (torch.cude.is_available()) else 'cpu')

    return device

# Defining a class that for the model


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']
        # Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        # model will contain the first 29 layers
        self.model = models.vgg19(pretrained=True).features[:29]

    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self, x):
        # initialize an array that wil hold the activations from the chosen layers
        features = []
        # Iterate over all the layers of the mode
        for layer_num, layer in enumerate(self.model):
            # activation of the layer will stored in x
            x = layer(x)
            # appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)

        return features

# Core of the transfer system


def train_transfer_image_style(source, target, outfile, epoch=250):
    # Loading the original and the style image
    content_image = image_loader(target)
    style_image = image_loader(source)

    show_images([source, target],
                titles=['Content image', 'Style image'])

    # Creating the generated image from the original image
    generated_image = content_image.clone().requires_grad_(True)

    # Load the model to the GPU
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
        total_loss = calculate_loss(gen_features, orig_feautes, style_featues)
        # optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print the image and save it after each 100 epoch
        if (e/100):
            save_image(generated_image, outfile)

    plt.imshow(mpimg.imread(outfile))


device = check_device()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="style source image name")
    parser.add_argument("--target", type=str, help="image to change style of")
    parser.add_argument("--outfile", type=str,
                        help="image output file path and name")
    parser.add_argument("--epochs", type=int,
                        help="number of style transfer cycles", default=250)

    args = parser.parse_args()

    train_transfer_image_style(
        args.source, args.target, args.outfile, args.epochs)

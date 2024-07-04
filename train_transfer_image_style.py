import matplotlib.pylab as plt
import matplotlib.image as mpimg
import torch.optim as optim

import utils as utils

from torchvision.utils import save_image

from tqdm import tqdm

from vgg_nets import VGG, VGG16, VGG19

device = utils.check_device()

# initialize the paramerters required for fitting the model
lr = 0.004
alpha = 8
beta = 70

# Core of the transfer system


def train_transfer_image_style(source, target, outfile, epoch=250, model_name='vgg', width=512, height=512):
    # Loading the original and the style image
    content_image = utils.image_loader(target, device, width, height)
    style_image = utils.image_loader(source, device, width, height)

    # Creating the generated image from the original image
    generated_image = content_image.clone().requires_grad_(True)

    # Load the model to the GPU
    model = VGG().to(device).eval()
    if model_name == 'vgg16':
        model = VGG16().to(device).eval()
    elif model_name == 'vgg19':
        model = VGG19().to(device).eval()

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

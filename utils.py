import torch
import platform

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms

from PIL import Image


def show_images(images, titles=('',), dim=1024):
    n = len(images)
    for i in range(n):
        img = mpimg.imread(images[i])
        plt.imshow(img, aspect='equal')
        plt.title(titles[i] if len(titles) > i else '')
        plt.show()


def image_loader(path, device):
    image = Image.open(path)
    # defining the image transformation steps to be performed before feeding them to the model
    loader = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()])
    # The preprocessing steps involves resizing the image and then converting it to a tensor
    image = loader(image.convert('RGB')).unsqueeze(0)
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


def calculate_loss(gen_features, orig_feautes, style_featues, alpha, beta):
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

    device = None

    if platform.system() == 'Darwin':
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if (torch.cude.is_available()) else 'cpu')

    return device

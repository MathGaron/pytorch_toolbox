from PIL import Image
from sklearn import manifold
from torch.autograd import Variable
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from examples.classification.cat_dog_net import CatDogNet
from pytorch_toolbox.transformations.image import Resize, Normalize, NumpyImage2Tensor
from pytorch_toolbox.transformations.to_float import ToFloat
from pytorch_toolbox.transformations.compose import Compose


def random_rect(n):
    rect = np.random.uniform(0, 1, (n, 2)).repeat(2, axis=1)
    hw = np.random.uniform(0.01, 0.5, (n, 2))
    rect[:, 1::2] += hw
    rect = np.clip(rect, 0, 1)
    return rect


def convert_image(image):
    imagenet_mean = [123, 116, 103]
    imagenet_std = [58, 57, 57]
    transformations = Compose([Resize((128, 128)),
                               ToFloat(),
                               NumpyImage2Tensor(),
                               Normalize(mean=imagenet_mean, std=imagenet_std)])
    return transformations(image).unsqueeze(0)


def generate_occluded_images(base_image, n_images, n_occluder=10):
    image_batch = np.zeros((n_images, base_image.shape[0], base_image.shape[1], base_image.shape[2]))
    image_batch[:, :, :, :] = base_image
    for i in tqdm(range(n_images)):
        rects = random_rect(n_occluder)
        rects[:, :2] *= base_image.shape[1]
        rects[:, 2:] *= base_image.shape[0]
        rects = rects.astype(np.int)
        for rect in rects:
            image_batch[i, rect[2]:rect[3], rect[0]:rect[1], :] = 0
    return image_batch

if __name__ == '__main__':

    backend = "cpu"     # cpu|cuda
    model_path = "cat_dog_model.pth.tar"

    #
    #   Instantiate model and load checkpoint
    #
    model = CatDogNet()
    model.load(model_path)
    model.eval()
    cat_path = "images/cat.jpg"
    dog_path = "images/dog.jpg"

    # Load a cat and a dog test images
    n_occluder = 10
    n_images = 200
    cat_img_numpy = np.array(Image.open(cat_path).convert('RGB'))
    dog_img_numpy = np.array(Image.open(dog_path).convert('RGB'))
    cat_images = generate_occluded_images(cat_img_numpy, n_images, n_occluder)
    dog_images = generate_occluded_images(dog_img_numpy, n_images, n_occluder)

    # Generate multiple version of both images with random occluders
    # input tensor are batch x Channel x 128 x 128
    batch = torch.FloatTensor(n_images*2, 3, 128, 128)
    for i in range(n_images):
        batch[i, :, :, :] = convert_image(cat_images[i, :, :, :])
        batch[i + n_images, :, :, :] = convert_image(dog_images[i, :, :, :])

    # Compute predictions/embeddings
    batch = Variable(batch)
    predictions = model(batch).data.cpu().numpy()

    # show distribution of predictions for both classes
    plt.subplot("121")
    plt.title("Cat predictions distribution")
    plt.hist(predictions[:n_images, 0])
    plt.subplot("122")
    plt.title("Dog predictions distribution")
    plt.hist(predictions[n_images:, 1])
    plt.show()

    # Compute 2D representation of the embeddings
    activations = model.load_activations()["lin1"]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X = tsne.fit_transform(activations)

    # Normalize
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    activations_2D = (X - x_min) / (x_max - x_min)

    plt.scatter(X[:n_images, 0], X[:n_images, 1], color="red")
    plt.scatter(X[n_images:, 0], X[n_images:, 1], color="blue")
    plt.legend(["cat", "dog"])
    plt.show()



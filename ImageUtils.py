import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training, predicting=False):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """

    if predicting:
        image = record.reshape((32, 32, 3))
    
    else:
        # Reshape from vector to [depth, height, width].
        depth_major = record.reshape((3, 32, 32))

        # Convert from [depth, height, width] to [height, width, depth]
        image = np.transpose(depth_major, [1, 2, 0])

    # Preprocess the image (if any)
    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training=0):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """

    if training:
        # Resize the image to add four extra pixels on each side.
        image = np.pad(image, ((4,4), (4,4), (0,0)))

        # Randomly crop a [32, 32] section of the image.
        x = np.random.randint(9)
        y = np.random.randint(9)
        image = image[x:x+32, y:y+32]

        # Randomly flip the image horizontally.
        image = image if np.random.randint(2) == 0 else np.fliplr(image)

        # Randomly flip the image vertically.
        image = image if np.random.randint(2) == 0 else np.flipud(image)

    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean)/std

    return image

def show_image(record, predicting=False):
    image = parse_record(record, 0, predicting)
    image = np.transpose(image, [1, 2, 0])
    plt.imshow(image)
    plt.show()

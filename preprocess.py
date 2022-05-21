import os
from imageio import imread, imwrite
from PIL import Image, ImageOps
from constants import img_height, img_width

is_burnable = {
    'paper': True,
    'cardboard': True,
    'plastic': True,
    'trash': True,
    'glass': False,
    'metal': False,
}


def preprocess_images_from_dir(from_directory, to_directory):
    '''
    copies and process images
    '''

    for subdir, dirs, files in os.walk(from_directory):
        for file in files:
            if len(file) <= 4 or file[-4:] != '.jpg':
                continue

            image = Image.open(os.path.join(subdir, file))

            # Apply Grayscale
            # image = ImageOps.grayscale(image)

            # Resize image
            image = image.resize((img_width, img_height))

            imwrite(os.path.join(to_directory, file), image)


def preprocess():
    '''
    preprocess gathered images into burnable and nonburnable folders
    '''

    dataset_original_path = os.path.join(os.getcwd(), 'dataset_original')
    dataset_output_path = os.path.join(os.getcwd(), 'dataset')

    try:
        os.makedirs(dataset_output_path)
        os.makedirs(os.path.join(dataset_output_path, 'burnable'))
        os.makedirs(os.path.join(dataset_output_path, 'nonburnable'))
    except OSError:
        if not os.path.isdir(dataset_output_path):
            raise

    for original_class in is_burnable:

        original_class_path = os.path.join(dataset_original_path,
                                           original_class)

        preprocessed_class_path = os.path.join(
            dataset_output_path,
            'burnable' if is_burnable[original_class] else 'nonburnable')

        preprocess_images_from_dir(original_class_path,
                                   preprocessed_class_path)

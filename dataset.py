import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from constants import img_height, img_width
# Parameters
batch_size = 32
validation_split = 0.3  # Train-test split. Validation = Test

print('Importing dataset...')
data_dir = pathlib.Path('dataset')

# Get the number of images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"image_count = {image_count}")

#
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print('Found classes ', train_ds.class_names)

# Use buffered prefetching
print('Using buffered prefetching...')
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# importing the libraries
import tensorflow as tf
import os
from keras import layers, models
import sys


def parse_image_mask(img_path, mask_path):
    # loading images and masks
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    # normalizing images and masks
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

def conv_block(x, filters):
    # the conv block consist of 2 convolution blocks
    x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation="relu"
    )(x)
    x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation="relu"
    )(x)
    return x

def build_unet(input_shape=(256,256,3)):
    # building the U_net model (c stands for conv and p for maxPooling)
    inputs = layers.Input(input_shape)

    # encoder level 1
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    # encoder level 2
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    # encoder level 3
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    # bootleneck
    b = conv_block(p3, 512)
    
    # decoder level 1
    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 256)

    # decoder level 2
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 128)

    # decoder level 3
    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 64)

    # output layer (sigmoid for binary recognition)
    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid"
    )(c6)

    return models.Model(inputs, outputs)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # converting y to float
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # the intersection take the total sum of pixels correctly predicted (y_true * y_pred gives the common pixels)
    # reduce_sum gives the total sum
    intersection = tf.reduce_sum(y_true * y_pred)
    
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice
    


# loading directories
PATCH_IMG_DIR = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/lazy_patches/images"
PATCH_MASK_DIR = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/lazy_patches/masks"

# creating a list containing all the image patches names and adding for each one the directory path
images_path = sorted([
    os.path.join(PATCH_IMG_DIR, fname)
    for fname in os.listdir(PATCH_IMG_DIR)
])


# creating a list containing all the mask patches names and adding for each one the directory path
masks_path = sorted([
    os.path.join(PATCH_MASK_DIR, fname)
    for fname in os.listdir(PATCH_MASK_DIR)
])


IMG_SIZE = 256

# creating a list of couples (image_path, mask_path)
dataset = tf.data.Dataset.from_tensor_slices(
    (images_path, masks_path)
)

# image and masks decoding and normalization
dataset = dataset.map(
    parse_image_mask,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.shuffle(buffer_size=1000)

# batch and prefetch
BATCH_SIZE = 8
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# dataset splitting (train and validation)
TOTAL = len(images_path)
VAL_SIZE = int(0.2 * TOTAL)
val_dataset = dataset.take(VAL_SIZE // BATCH_SIZE)
train_dataset = dataset.skip(VAL_SIZE // BATCH_SIZE)


# model building and compiling

model = build_unet(input_shape=(256, 256, 3))
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
epochs = 10

# compiling model
model.compile(
    optimizer=optimizer,
    loss=bce_dice_loss,
    metrics=[dice_coefficient]
)

# saving model architecture
with open("/workspace/progetto_dati_satellitari/object_detection/buildings/model_architecture.txt", "w") as f:
    old_stdout = sys.stdout
    sys.stdout = f
    model.summary()
    sys.stdout = old_stdout


# model training
import matplotlib.pyplot as plt
history = model.fit(train_dataset, validation_data = val_dataset, epochs = epochs)


# plotting model results
print(history.history.keys())

import matplotlib.pyplot as plt
# interpreting the plotting:
# dice train and val need to rise and the loss to fall
# excessive rise of dice train might indicate overfitting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.53, 11.69))
ax1.plot(history.history['dice_coefficient'])
ax1.plot(history.history['val_dice_coefficient'])
ax1.set_xlabel('epoch')
ax1.set_ylabel('dice coefficient')
ax1.set_title('Dice Coefficient over Epochs')
ax1.legend(['Train', 'Validation'], loc='lower right')
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.set_title('Loss over Epochs')
ax2.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(f"/workspace/progetto_dati_satellitari/object_detection/buildings/model_and_tools/Trained_model_LAZY_{epochs}_epochs.jpg", dpi=600)
plt.close()

model.save(f"/workspace/progetto_dati_satellitari/object_detection/buildings/model_and_tools/Trained_model_LAZY_{epochs}_epochs") 

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pandas as pd
import os

# Capsule Layer
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.kernel = self.add_weight(name='capsule_kernel',
                                      shape=(input_shape[-1], self.num_capsules * self.dim_capsule),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs):
        u_hat = tf.linalg.matmul(inputs, self.kernel)
        u_hat = tf.reshape(u_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsule))
        u_hat = tf.transpose(u_hat, perm=[0, 2, 1, 3])
        for i in range(self.routings):
            c = tf.nn.softmax(tf.reduce_sum(u_hat, axis=-1, keepdims=True), axis=2)
            outputs = tf.linalg.matmul(c, u_hat, transpose_a=True)
        return tf.sqrt(tf.reduce_sum(tf.square(outputs), axis=-1))

# Length Layer (for output probabilities)
class Length(layers.Layer):
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))

# Capsule Network Model
def create_capsule_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)

    # Conv Layer
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Capsule Layer
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Reshape((-1, 128))(x)
    capsules = CapsuleLayer(num_capsules=n_classes, dim_capsule=16, routings=3)(x)

    # Output Layer
    output = Length()(capsules)

    model = models.Model(inputs=inputs, outputs=output)
    return model

# Kaggle-specific setup
if __name__ == "__main__":
    # Path to HAM10000 dataset files
    metadata_path = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
    images_dir1 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/"
    images_dir2 = "/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/"

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Combine image paths
    metadata['image_path'] = metadata['image_id'].apply(
        lambda x: images_dir1 + x + ".jpg" if os.path.exists(images_dir1 + x + ".jpg") else images_dir2 + x + ".jpg"
    )

    # Encode labels
    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['dx'])

    # Load images and preprocess
    images = []
    labels = []

    for _, row in metadata.iterrows():
        img = load_img(row['image_path'], target_size=(224, 224))  # Resize to match model input
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
        labels.append(row['label'])

    # Convert to NumPy arrays
    images = np.array(images)
    labels = to_categorical(labels, num_classes=7)  # One-hot encode labels

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Shuffle and split the data
    images, labels = shuffle(images, labels, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Create model
    input_shape = (224, 224, 3)  # Adjust for HAM10000 image sizes
    n_classes = 7  # Number of skin cancer types
    model = create_capsule_model(input_shape, n_classes)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        datagen.flow(train_data, train_labels, batch_size=32),
        validation_data=(val_data, val_labels),
        epochs=50,  # Increased epochs for better performance
        steps_per_epoch=len(train_data) // 32
    )

    # Save the model
    model.save("capsule_model.h5")

    print("Model training completed and saved as capsule_model.h5")

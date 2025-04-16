import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

# ------------------------------------------------------
# PARAMETERS & DIRECTORY SETUP
# ------------------------------------------------------
# Use a raw string to avoid escape sequence issues on Windows.
dataset_dir = r"C:\Users\benny\Sonogram\dataset"  # Folder containing subfolders: normal, benign, malignant
classes = ['normal', 'benign', 'malignant']
# Map class names to numeric labels (0, 1, 2)
class_to_label = {"normal": 0, "benign": 1, "malignant": 2}

# Image dimensions and training hyperparameters
img_width, img_height = 150, 150  # Adjust dimensions as needed
batch_size = 16
epochs = 30
learning_rate = 1e-4

# ------------------------------------------------------
# FUNCTION TO GET FILE PAIRS
# ------------------------------------------------------
def get_image_mask_pairs(class_dir):
    """
    Returns a list of tuples (primary_image_path, mask_image_path) for the given folder.
    Assumes primary image files do not contain "mask" in their filename and that the corresponding
    mask file is named with the same base name appended with "_mask" before the extension.
    """
    pairs = []
    # Print the directory being checked for debugging
    print("Checking directory:", class_dir)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return pairs
    for fname in os.listdir(class_dir):
        lower_fname = fname.lower()
        if "mask" in lower_fname:
            continue  # Skip mask files when listing primary images
        primary_path = os.path.join(class_dir, fname)
        base, ext = os.path.splitext(fname)
        # Construct expected mask filename
        mask_fname = f"{base}_mask{ext}"
        mask_path = os.path.join(class_dir, mask_fname)
        if os.path.exists(mask_path):
            pairs.append((primary_path, mask_path))
        else:
            print(f"Warning: No mask found for {primary_path}")
    return pairs

# ------------------------------------------------------
# BUILD LISTS OF FILE PAIRS AND LABELS FOR THE ENTIRE DATASET
# ------------------------------------------------------
all_pairs = []
all_labels = []
for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    pairs = get_image_mask_pairs(cls_dir)
    all_pairs.extend(pairs)
    all_labels.extend([class_to_label[cls]] * len(pairs))

print("Total samples found:", len(all_pairs))

# ------------------------------------------------------
# SPLIT DATA INTO TRAIN, VALIDATION, AND TEST SETS
# ------------------------------------------------------
data = list(zip(all_pairs, all_labels))
np.random.shuffle(data)
all_pairs_shuffled, all_labels_shuffled = zip(*data)

n_total = len(all_pairs_shuffled)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
# Remaining samples for test
n_test = n_total - n_train - n_val

train_pairs = all_pairs_shuffled[:n_train]
train_labels = all_labels_shuffled[:n_train]

val_pairs = all_pairs_shuffled[n_train:n_train+n_val]
val_labels = all_labels_shuffled[n_train:n_train+n_val]

test_pairs = all_pairs_shuffled[n_train+n_val:]
test_labels = all_labels_shuffled[n_train+n_val:]

print(f"Train samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}, Test samples: {len(test_pairs)}")

# ------------------------------------------------------
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ------------------------------------------------------
def load_image_pair(primary_path, mask_path, label):
    """
    Reads and preprocesses an image and its mask.
    - Decodes JPEG images.
    - Resizes them to [img_height, img_width].
    - Normalizes pixel values to [0,1].
    - Concatenates the primary image (3 channels) and the mask (1 channel) 
      to create a 4-channel input.
    """
    # Load and process primary image
    primary_img = tf.io.read_file(primary_path)
    primary_img = tf.image.decode_jpeg(primary_img, channels=3)
    primary_img = tf.image.resize(primary_img, [img_height, img_width])
    primary_img = primary_img / 255.0  # Normalize to [0, 1]

    # Load and process mask image (assumed to be single channel)
    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=1)
    mask_img = tf.image.resize(mask_img, [img_height, img_width])
    mask_img = mask_img / 255.0

    # Concatenate primary image and mask along the channel axis -> shape: (img_height, img_width, 4)
    image = tf.concat([primary_img, mask_img], axis=-1)
    return image, label

def create_dataset(pairs, labels, batch_size=batch_size, shuffle=True):
    """
    Creates a tf.data.Dataset from lists of (primary, mask) file pairs and their labels.
    """
    primary_paths = [pair[0] for pair in pairs]
    mask_paths = [pair[1] for pair in pairs]
    ds = tf.data.Dataset.from_tensor_slices((primary_paths, mask_paths, list(labels)))
    ds = ds.map(load_image_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(primary_paths))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

# Create datasets for training, validation, and testing
train_dataset = create_dataset(train_pairs, train_labels, batch_size, shuffle=True)
val_dataset = create_dataset(val_pairs, val_labels, batch_size, shuffle=False)
test_dataset = create_dataset(test_pairs, test_labels, batch_size, shuffle=False)

# ------------------------------------------------------
# MODEL DEFINITION: SIMPLE CNN ACCEPTING 4-CHANNEL INPUT
# ------------------------------------------------------
model = Sequential([
    Input(shape=(img_height, img_width, 4)),  # Explicitly define Input layer; 4 channels (3 for image, 1 for mask)
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Three classes: normal, benign, malignant
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',  # Using integer labels
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# TRAINING THE MODEL
# ------------------------------------------------------
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset
)

# ------------------------------------------------------
# FINAL EVALUATION ON TEST DATA
# ------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy: {:.4f}".format(test_accuracy))

# ------------------------------------------------------
# SAVE THE TRAINED MODEL
# ------------------------------------------------------
model.save("sonogram_tumor_detection_model.h5")
print("Model saved as sonogram_tumor_detection_model.h5")

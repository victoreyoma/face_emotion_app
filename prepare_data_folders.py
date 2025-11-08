# prepare_data_folders.py
# Usage: python prepare_data_folders.py --data data --out data/fer2013_prepared_folders.npz

import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_fer2013_from_folders(data_dir, img_size=(48, 48)):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_gen = datagen.flow_from_directory(
        train_dir, target_size=img_size, color_mode='grayscale',
        class_mode='sparse', shuffle=True)
    test_gen = datagen.flow_from_directory(
        test_dir, target_size=img_size, color_mode='grayscale',
        class_mode='sparse', shuffle=False)

    # Load all train data into memory
    X_train, y_train = [], []
    for i in range(len(train_gen)):
        X, y = train_gen[i]
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Same for test data
    X_test, y_test = [], []
    for i in range(len(test_gen)):
        X, y = test_gen[i]
        X_test.append(X)
        y_test.append(y)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_gen.class_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='Path containing train/ and test/ folders')
    parser.add_argument('--out', type=str, default='data/fer2013_prepared_folders.npz', help='Output .npz file path')
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_indices = load_fer2013_from_folders(args.data)

    np.savez_compressed(args.out,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test,
                        class_indices=class_indices)
    print("✅ Saved:", args.out)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Class mapping:", class_indices)

if __name__ == '__main__':
    main()

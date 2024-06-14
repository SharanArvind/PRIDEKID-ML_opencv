import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import cv2

# Load and preprocess the data
df = pd.read_csv('fer2013/fer2013/fer2013.csv')

emotion_label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Filter only the emotions of interest
INTERESTED_LABELS = [3, 4, 6]  # happiness, sadness, neutral
df = df[df.emotion.isin(INTERESTED_LABELS)]

# Prepare image arrays and labels
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)
le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = to_categorical(img_labels)

X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels, shuffle=True, stratify=img_labels, test_size=0.1, random_state=42)

# Normalize image data
X_train = X_train / 255.0
X_valid = X_valid / 255.0

# Build the model
def build_net(optim):
    net = Sequential(name='DCNN')
    net.add(Conv2D(64, kernel_size=(5,5), input_shape=(48, 48, 1), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1'))
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(Conv2D(64, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2'))
    net.add(BatchNormalization(name='batchnorm_2'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))
    net.add(Conv2D(128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3'))
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(Conv2D(128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4'))
    net.add(BatchNormalization(name='batchnorm_4'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))
    net.add(Conv2D(256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5'))
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(Conv2D(256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6'))
    net.add(BatchNormalization(name='batchnorm_6'))
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))
    net.add(Flatten(name='flatten'))
    net.add(Dense(128, activation='elu', kernel_initializer='he_normal', name='dense_1'))
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Dropout(0.6, name='dropout_4'))
    net.add(Dense(3, activation='softmax', name='out_layer'))
    net.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    net.summary()
    return net

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=11, verbose=1, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-7, verbose=1)

callbacks = [early_stopping, lr_scheduler]

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.15, height_shift_range=0.15, shear_range=0.15, zoom_range=0.15, horizontal_flip=True)
train_datagen.fit(X_train)

batch_size = 32
epochs = 100
optimizer = optimizers.Adam(0.001)
model = build_net(optimizer)
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    use_multiprocessing=False  # Disable multiprocessing
)


model.save("model.h5")

# Plot training history
sns.set()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(history.epoch, history.history['accuracy'], label='train', ax=ax1)
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid', ax=ax1)
ax1.set_title('Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
sns.lineplot(history.epoch, history.history['loss'], label='train', ax=ax2)
sns.lineplot(history.epoch, history.history['val_loss'], label='valid', ax=ax2)
ax2.set_title('Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
plt.savefig('epoch_history_dcnn.png')
plt.show()

# Evaluate model performance
yhat_valid = model.predict_classes(X_valid)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))
plt.savefig("confusion_matrix_dcnn.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))

# Real-time emotion detection
def detect_face_and_predict_emotion(model, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_reshaped = face_normalized.reshape(1, 48, 48, 1)
        emotion_prediction = model.predict(face_reshaped)
        max_index = np.argmax(emotion_prediction[0])
        emotion = emotion_label_to_text[le.inverse_transform([max_index])[0]]
        label = "Attentive" if emotion in ["happiness", "neutral"] else "Not Attentive"
        color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# OpenCV to capture webcam video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_face_and_predict_emotion(model, frame)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

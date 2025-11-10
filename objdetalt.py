import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D

# Dataset path
root_dir = "caltech-101-img"

# 1. Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    root_dir, target_size=(224, 224), batch_size=32, 
    class_mode='categorical', subset='training')

val_gen = datagen.flow_from_directory(
    root_dir, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation')

num_classes = train_gen.num_classes

# 2. Load VGG16 and freeze layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_model.layers:
    layer.trainable = False

# 3. Build model with custom classifier
model = Sequential([
    vgg_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 4. Compile and train
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_gen, validation_data=val_gen, epochs=5)  # Reduced from 10

# 5. Fine-tune: unfreeze last 4 layers
for layer in vgg_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 6. Evaluate and save
loss, acc = model.evaluate(val_gen)
print(f"\nFinal Accuracy: {acc*100:.2f}%")
model.save('caltech101_model.h5')

# 7. Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Train')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Train')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()

# 8. Sample predictions
x_test, y_test = next(val_gen)
predictions = model.predict(x_test)
labels = list(train_gen.class_indices.keys())

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    pred = labels[np.argmax(predictions[i])]
    true = labels[np.argmax(y_test[i])]
    plt.title(f"P:{pred}\nT:{true}", fontsize=8)
plt.show()
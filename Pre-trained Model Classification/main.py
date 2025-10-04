from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1) Dataset paths adjust as per requirement
train_dir = "data/train"
val_dir = "data/val"
img_size = (224, 224)
batch_size = 32

# 2) Data generators
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

num_classes = train_gen.num_classes

# 3) Pre-trained model (ResNet50) without top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base model

# 4) Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 5) Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6) Train model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# 7) Evaluate model
loss, acc = model.evaluate(val_gen)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")
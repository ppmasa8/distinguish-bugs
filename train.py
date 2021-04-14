from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

def main():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    #model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        "Data/train",
        target_size=(64, 64),
        batch_size=10)
    validation_generator = test_datagen.flow_from_directory(
        "Data/validation",
        target_size=(64, 64),
        batch_size=10)
    model.fit_generator(
        train_generator,
        epochs=20,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=10)
    model.save("model.h5")
if __name__ == "__main__":
    main()
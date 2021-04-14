from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np

def main():
    model = Sequential()
    model.add(Dense(3, input_dim=2))
    model.add(Activation("sigmoid"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # model.summary()
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"])
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    model.fit(x, y, epochs=5000)
    result = model.predict(x)
    print(result)
if __name__ == "__main__":
    main()
import tensorflow as tf
import json
import matplotlib.pyplot as plt

label2num = {"wall": 0, "pin": 1, "screw": 2, "tetrahedron": 3, "ball": 4}
num2label = {0: "wall", 1: "pin", 2: "screw", 3: "tetrahedron", 4: "ball"}


class PoseClassifier:
    def __init__(self, input_shape=(17, 3), learning_rate=0.001):
        self.model = self.create_classifier_model(input_shape)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def create_classifier_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.5))  # Add dropout for regularization
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(5, activation="softmax"))
        return model

    def train(
        self, train_data, train_labels, validation_data, epochs=10, batch_size=32
    ):
        checkpoint_path = "pose_classifier.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        history = self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=[checkpoint_callback],
        )
        return history

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

    def predict(self, data):
        return self.model.predict(data)

    @classmethod
    def load_model(cls, file_path, input_shape):
        loaded_model = cls(input_shape)
        loaded_model.model = tf.keras.models.load_model(file_path)
        return loaded_model


class KeypointDataset:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        (
            self.train_data,
            self.train_labels,
            self.val_data,
            self.val_labels,
        ) = self.split_dataset()

    def load_data(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    def split_dataset(self):
        data_list = []
        label_list = []

        for entry in self.data.values():
            label_list.append(label2num[entry["label"]])
            data_list.append(entry["keypoints_with_scores"])

        data = tf.convert_to_tensor(data_list, dtype=tf.float32)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        # Split the dataset into train and validation sets (90% train, 10% validation)
        split_idx = int(0.8 * len(data))
        train_data, val_data = tf.split(data, [split_idx, len(data) - split_idx])
        train_labels, val_labels = tf.split(
            labels, [split_idx, len(labels) - split_idx]
        )

        return train_data, train_labels, val_data, val_labels


def plot_training_history(history):
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load data
    dataset = KeypointDataset("pose_images/keypoints.json")

    # Create and train the classifier
    classifier = PoseClassifier(input_shape=(17, 3), learning_rate=0.0005)
    # Train with tqdm progress bar
    history = classifier.train(
        dataset.train_data,
        dataset.train_labels,
        validation_data=(dataset.val_data, dataset.val_labels),
        epochs=200,
        batch_size=16,
    )

    # Plot training history
    plot_training_history(history)

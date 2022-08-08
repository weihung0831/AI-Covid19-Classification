import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, \
    confusion_matrix
from tensorflow.keras.models import load_model


def test_model(test_img, y_true, class_name, model_path, confusion_matrix_plot_path):
    """
    The test_model function is used to test the model on a given image.

    :param test_img: Pass the test image to be used for testing
    :param y_true: Pass the true labels of the test set
    :param class_name: Display the labels in the confusion matrix
    :param model_path: Specify the path of the model to be loaded
    :param confusion_matrix_plot_path: Save the confusion matrix plot
    :return: The accuracy score and classification report of the model on a given test image
    """

    model = load_model(model_path)
    y_pred = model.predict(test_img)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    print(accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_name))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=class_name
    ).plot()
    plt.savefig(confusion_matrix_plot_path)


if __name__ == "__main__":
    data = np.load('./dataset/data.npz')
    x_test, y_test = (
        data["test_img"],
        data["test_label"],
    )
    print(x_test.shape, y_test.shape)

    class_name = ["covid", "normal", "pneumonia"]
    test_model(
        x_test,
        y_test,
        class_name,
        model_path="./model/model.tf/",
        confusion_matrix_plot_path="./model/confusion_matrix.png",
    )

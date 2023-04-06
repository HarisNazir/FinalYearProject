import numpy as np
from classifiers import *
from pipeline import *
import json
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

with tf.device('/GPU:0'):
    def loss_matrix(y_true, y_pred):
        return 1.0 - accuracy(y_true, y_pred)

    def accuracy(y_true, y_pred):
        return np.mean(np.equal(y_true, np.round(y_pred)))

    with open('metadata.json') as f: #JSON HERE
        ground_truth = json.load(f)
    # 1 - Load the model and its pretrained weights
    classifier = Meso4()
    classifier.load('weights/Meso4_DF.h5')
    
    videos_predictions_correct = 0




    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

    # 2 - Predict
    X, y = generator.next()
    print('Predicted :', classifier.predict(X), '\nReal class :', y)

    # 3 - Prediction for a video dataset

    classifier.load('weights/Meso4_F2F.h5')


    predictions = compute_accuracy(classifier, 'data_split/Validation')

    # Get the ground truth labels
    ground_truth_labels = []
    for video_name, prediction in predictions.items():
        video_name = video_name + '.mp4'
        ground_truth_value = ground_truth[video_name]['label']
        ground_truth_labels.append(1.0 if ground_truth_value == "FAKE" else 0.0)
        predicted_label = 1.0 if prediction[0] >= 0.5 else 0.0
        ground_truth_label = 0.0 if ground_truth_value == "REAL" else 1.0
        print("Video: ", video_name)
        print("Ground Truth Label: ", ground_truth_label)
        print("Predicted Label: ", predicted_label)

    # Get the predicted labels
    predicted_labels = [prediction[0] for prediction in predictions.values()]

    ground_truth_labels = [1 if label == "FAKE" else 0 for label in ground_truth_labels]
    predicted_labels = [1 if prediction >= 0.5 else 0 for prediction in predicted_labels]

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(predicted_labels, ground_truth_labels)

    # Visualize the confusion matrix
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])
    plt.colorbar()
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='black')
    plt.show()

    # Calculate the loss rate
    loss_rate = loss_matrix(ground_truth_labels, predicted_labels)

    # Calculate the accuracy rate
    accuracy_rate = accuracy(ground_truth_labels, predicted_labels)

    # Calculate the accuracy and loss rate values for each video
    # To do: ACCURACY NEEDS TO BE FLIPPED
    accuracy_values = []
    loss_rate_values = []
    for video_name, prediction in predictions.items():
        video_name = video_name + '.mp4'
        ground_truth_value = ground_truth[video_name]['label']
        ground_truth_label = 1.0 if ground_truth_value == "REAL" else 0.0
        predicted_label = 1 if prediction[0] >= 0.5 else 0
        accuracy_value = accuracy([ground_truth_label], [predicted_label])
        loss_rate_value = loss_matrix([ground_truth_label], [predicted_label])
        accuracy_values.append(accuracy_value)
        loss_rate_values.append(loss_rate_value)

    # Calculate the average accuracy and loss rate values
    average_accuracy = np.mean(accuracy_values)
    print("Accuracy rate: {:.2f}%".format(average_accuracy * 100))
    print("Average Accuracy: ", average_accuracy)

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import argparse
import json


def main():
    # Ορισμός των παραμέτρων από τη γραμμή εντολών
    parser = argparse.ArgumentParser(description='K-NN Classifier')
    parser.add_argument('--training_path', type=str, help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, help='Path to the test CSV file')
    parser.add_argument('--output_file', type=str, help='Output JSON file name')
    args = parser.parse_args()

    # Χρήση των ορισμάτων από τη γραμμή εντολών για τα αρχεία CSV
    file_path1 = args.training_path
    file_path2 = args.test_path
    output_file = args.output_file
    # Ο υπόλοιπος κώδικας παραμένει όπως ήταν

    # Load CSV files
    data_training = pd.read_csv(file_path1)
    data_test = pd.read_csv(file_path2)


    # Undersample for a balanced dataset
    indices_no = data_training.index[data_training['REFACTORED'] == 0]
    indices_yes = data_training.index[data_training['REFACTORED'] == 1]

    le = min(len(indices_yes), len(indices_no))
    np.random.seed(12)
    indices_no_undersampled = np.random.choice(indices_no, le, replace=False)
    indices_yes_undersampled = np.random.choice(indices_yes, le, replace=False)

    data_balanced = pd.concat([data_training.loc[indices_yes_undersampled], data_training.loc[indices_no_undersampled]])
    data_balanced = shuffle(data_balanced, random_state=1234)

    # Remove columns
    data_training_processed = data_balanced.iloc[:, 3:]
    data_training_processed['REFACTORED'] = data_training_processed['REFACTORED'].astype('category')

    data_test_processed = data_test.iloc[:, 3:]
    data_test_processed['REFACTORED'] = data_test_processed['REFACTORED'].astype('category')


    X_train = data_training_processed.drop('REFACTORED', axis=1)
    y_train = data_training_processed['REFACTORED']
    X_test = data_test_processed.drop('REFACTORED', axis=1)
    y_test = data_test_processed['REFACTORED']


    # Define the range of values for k, distance, and weights
    k_values = list(range(1, 21))
    distance_metrics = [1, 2, 3, 4, 5]  # 1 for Manhattan distance, 2 for Euclidean distance
    weights_options = ['distance']


    best_f1 = 0
    best_params = {}

    # Iterate over parameter combinations
    for k_value in k_values:
        for distance_metric in distance_metrics:
            for weights_option in weights_options:
                # Train k-NN model
                knn_model = KNeighborsClassifier(n_neighbors=k_value, p=distance_metric, weights=weights_option)
                knn_model.fit(X_train, y_train)

                # Make predictions on test data
                y_pred = knn_model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Print or store the results
                print(f'k={k_value}, distance={distance_metric}, weights={weights_option}: f1={f1:.2f}')

                # Update the best parameters if the current model has higher accuracy
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'k': k_value, 'distance': distance_metric, 'weights': weights_option}

    # Αποθήκευση των αποτελεσμάτων σε ένα αρχείο JSON
    print(f'Best parameters: {best_params}, Best F1: {best_f1:.2f}')
    result_data = {'Best Parameters': best_params, 'Best F1': best_f1}
    with open(output_file, 'w') as json_file:
        json.dump(result_data, json_file)

if __name__ == '__main__':
    main()
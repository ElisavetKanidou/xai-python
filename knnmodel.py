import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import argparse
import pandas as pd
import pickle
import json


def main():
    parser = argparse.ArgumentParser(description='K-NN Classifier')
    parser.add_argument('--training_path', type=str, help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, help='Path to the test CSV file')
    parser.add_argument('--output_file', type=str, help='Output JSON file name')
    parser.add_argument('--k_value', type=int, help='k_value')
    parser.add_argument('--distance_metric', type=int, help='distance_metric')

    args = parser.parse_args()

    # Χρήση των ορισμάτων από τη γραμμή εντολών για τα αρχεία CSV
    file_path1 = args.training_path
    file_path2 = args.test_path
    output_file = args.output_file
    k_value = args.k_value
    distance_metric = args.distance_metric

    # Διαβάστε τα αρχεία CSV
    data_training = pd.read_csv(file_path1)
    data_test = pd.read_csv(file_path2)

    # Υποδείγμα (undersampling) για ισορροπημένο σύνολο δεδομένων
    indices_no = data_training.index[data_training['REFACTORED'] == 0]
    indices_yes = data_training.index[data_training['REFACTORED'] == 1]

    le = min(len(indices_yes), len(indices_no))
    np.random.seed(12)
    indices_no_undersampled = np.random.choice(indices_no, le, replace=False)
    indices_yes_undersampled = np.random.choice(indices_yes, le, replace=False)

    data_balanced = pd.concat([data_training.loc[indices_yes_undersampled], data_training.loc[indices_no_undersampled]])
    data_balanced = shuffle(data_balanced, random_state=1234)

    # Καθορισμός του ονόματος του αρχείου CSV
    csv_file_path = output_file + '_resultdata_training.csv'

    # Αποθήκευση του DataFrame στο αρχείο CSV
    data_balanced.to_csv(csv_file_path, index=False)

    # Αφαίρεση στηλών
    data_training_processed = data_balanced.iloc[:, 3:]  # Αφαίρεση των πρώτων 4 στηλών

    data_training_processed['REFACTORED'] = data_training_processed['REFACTORED'].astype('category')

    data_test_processed = data_test.iloc[:, 3:]  # Αφαίρεση των πρώτων 4 στηλών
    data_test_processed['REFACTORED'] = data_test_processed['REFACTORED'].astype('category')

    # Εκπαίδευση μοντέλου k-NN

    X_train = data_training_processed.drop('REFACTORED', axis=1)
    y_train = data_training_processed['REFACTORED']
    X_test = data_test_processed.drop('REFACTORED', axis=1)
    y_test = data_test_processed['REFACTORED']

    # Εκπαίδευση μοντέλου k-NN
    knn_model = KNeighborsClassifier(n_neighbors=k_value, p=distance_metric, weights='distance')
    knn_model.fit(X_train.values, y_train.values)

    y_pred = knn_model.predict(X_test.values)

    # Υπολογισμός του accuracy και του F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1_sco = f1_score(y_test, y_pred, average='weighted')
    # print(f1)
    # Υπολογισμός του confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Εμφάνιση του confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()

    feature_names = list(data_test_processed.columns[:-1])

    # Προσθήκη ονομάτων χαρακτηριστικών στο μοντέλο

    knn_model.feature_names = feature_names
    # Αποθήκευση του μοντέλου με τα ονόματα των χαρακτηριστικών
    model_name = output_file + "_knn_model.pkl"
    with open(model_name, 'wb') as model_file:
        pickle.dump(knn_model, model_file)

    # Δημιουργία ενός λεξικού για τα αποτελέσματα
    results = {
        "accuracy": accuracy,
        "f1_score": f1_sco
    }
    #print(results)

    # Μετατροπή σε JSON string
    json_string = json.dumps(results)

    # Καθορισμός του ονόματος του αρχείου
    file_name = "results.json"

    # Αποθήκευση του JSON string στο αρχείο
    with open(file_name, "w") as json_file:
        json_file.write(json_string)

if __name__ == '__main__':
    main()

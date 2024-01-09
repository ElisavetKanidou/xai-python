import argparse
import pandas as pd
import pickle


def main():
    parser = argparse.ArgumentParser(description='K-NN Classifier')
    parser.add_argument('--predict_path', type=str, help='Path to the predict CSV file')
    parser.add_argument('--output_file', type=str, help='Output JSON file name')
    parser.add_argument('--knnmodel', type=str, help='knn model')

    args = parser.parse_args()

    # Χρήση των ορισμάτων από τη γραμμή εντολών για τα αρχεία CSV
    predict_path = args.predict_path
    output_file = args.output_file
    knn_model = args.knnmodel

    # Διαβάστε τα αρχεία CSV
    data_test = pd.read_csv(predict_path)

    # Φόρτωση του μοντέλου από το αρχείο
    with open(knn_model, 'rb') as model_file:
        knn_model = pickle.load(model_file)

    data_test_processed = data_test.iloc[:, 3:]  # Αφαίρεση των πρώτων 4 στηλών
    data_test_processed['REFACTORED'] = data_test_processed['REFACTORED'].astype('category')

    X_test = data_test_processed.drop('REFACTORED', axis=1)

    # Πρόβλεψη μοντέλου k-NN

    y_pred = knn_model.predict(X_test.values)
    y_proba = knn_model.predict_proba(X_test.values)

    data_test = data_test.drop('REFACTORED', axis=1)
    # print(y_proba)
    # Δημιουργία DataFrame με τα αποτελέσματα
    results_df = pd.concat([pd.DataFrame(data_test), pd.Series(y_pred, name='Predicted'),
                            pd.DataFrame(y_proba, columns=['Probability_0', 'Probability_1'])], axis=1)

    # Εκτύπωση αποτελεσμάτων σε μορφή JSON
    result_json = results_df[['file', 'Predicted', 'Probability_0', 'Probability_1']].to_json(orient='records',
                                                                                              lines=True)

    # Αποθήκευση του JSON σε αρχείο
    file_name = output_file + "_output_predict.json"
    with open(file_name, 'w') as json_file:
        json_file.write(result_json)


if __name__ == '__main__':
    main()

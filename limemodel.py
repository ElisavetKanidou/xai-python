import argparse
import pickle
import pandas as pd
from lime import lime_tabular
def main():
    parser = argparse.ArgumentParser(description='K-NN Classifier')
    parser.add_argument('--training_path', type=str, help='Path to the training CSV file')
    parser.add_argument('--data_path', type=str, help='Path to the data CSV file')
    parser.add_argument('--selected', type=str, help='Selectes class')
    parser.add_argument('--output_file', type=str, help='Output JSON file name')
    parser.add_argument('--knnmodel', type=str, help='knn model')

    args = parser.parse_args()

    # Χρήση των ορισμάτων από τη γραμμή εντολών για τα αρχεία CSV
    file_pathtraining = args.training_path
    file_path = args.data_path
    selected = args.selected
    output_file = args.output_file
    knn_model = args.knnmodel

    data_training = pd.read_csv(file_pathtraining)
    data = pd.read_csv(file_path)


    # Φόρτωση του μοντέλου από το αρχείο
    with open(knn_model, 'rb') as model_file:
        knn_model = pickle.load(model_file)

    data_training = data_training.iloc[:, 3:-1]   # Αφαίρεση των πρώτων 3 στηλών
    #print(knn_model.feature_names)
    #print(data_training)
    # Χρήση του LIME για εξήγηση
    feature_names = list(data_training.columns)  # Υποθέτοντας ότι η τελευταία στήλη είναι η μεταβλητή-στόχος


    explainer = lime_tabular.LimeTabularExplainer(data_training.values, feature_names=feature_names, class_names=[0, 1], discretize_continuous=True)
    knn_model.feature_names = feature_names
    # Επιλέγετε τη γραμμή με βάση τη συγκεκριμένη συνθήκη
    selected_row = data[data['file'] == selected]
    #print(selected_row)
    # Εκτύπωση της επιλεγμένης γραμμής
    selected_row_for_explanation = selected_row.iloc[:, 3:-1]

    # Convert data type to float before explaining
    explanation = explainer.explain_instance(selected_row_for_explanation.values[0], knn_model.predict_proba)
    print("explanation fit: "+explanation.score)

    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 6)  # Set the figure size explicitly
    plotname = output_file + '_plot.png'
    fig.savefig(plotname, bbox_inches='tight')  # Use bbox_inches='tight' to avoid cutting off


if __name__ == '__main__':
    main()
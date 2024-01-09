import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description='K-NN Classifier')
    parser.add_argument('--training_path', type=str, help='Path to the training CSV file')
    parser.add_argument('--data_path', type=str, help='Path from selected')
    parser.add_argument('--selected', type=str, help='Selected class')
    parser.add_argument('--output_file', type=str, help='output file')
    parser.add_argument('--knnmodel', type=str, help='knn model')
    parser.add_argument('--param1', type=str, help='Metric')
    parser.add_argument('--param2', type=str, help='Metric')
    parser.add_argument('--param3', type=str, help='Metric')

    args = parser.parse_args()

    # Χρήση των ορισμάτων από τη γραμμή εντολών για τα αρχεία CSV
    file_path1 = args.training_path
    data_path = args.data_path
    selected = args.selected
    output_file = args.output_file
    knn_model = args.knnmodel
    param1 = args.param1
    param2 = args.param2
    param3 = args.param3

    # Διαβάστε τα αρχεία CSV
    data_training = pd.read_csv(file_path1)

    with open(knn_model, 'rb') as model_file:
        knn_model = pickle.load(model_file)

    data = pd.read_csv(data_path)


    # Επιλογή της σειράς με βάση το αρχείο 'AccessTokenServiceTest.java'
    selected_row_indices = np.where(data['file'] == selected)[0]
    datab = data.iloc[:, 3:]  # Αφαίρεση των πρώτων 4 στηλών

    # Επιστρέφει τους γείτονες μαζί με τις αποστάσεις
    distances, neighbors = knn_model.kneighbors(datab.iloc[selected_row_indices, :-1].values,
                                                return_distance=True)

    # Εύρεση των ευρετηρίων ταξινόμησης για τις αποστάσεις
    sorted_indices = np.argsort(distances)

    # Εφαρμογή της ίδιας σειράς ταξινόμησης στους γείτονες
    neighbors_sorted = neighbors[:, sorted_indices]
    distances_sorted = distances[:, sorted_indices]
    neighbors_sorted = neighbors_sorted[0]
    distances_sorted = distances_sorted[0]

    # print("Οι ταξινομημένοι γείτονες με τις αποστάσεις για κάθε παρατήρηση είναι:")
    # for i in range(neighbors.shape[0]):
    #    print(f"Γείτονες για παρατήρηση {i + 1}: {neighbors_sorted[i]}, Αποστάσεις: {distances_sorted[i]}")

    # Προσθήκη των γειτόνων στο DataFrame
    result_dfa = pd.DataFrame()
    for i in range(neighbors_sorted.shape[1]):
        test_data = data_training.iloc[neighbors_sorted[:, i], :].copy()  # Δημιουργία αντιγράφου
        test_data.loc[:, 'Distance'] = distances_sorted[:, i]  # Χρήση .loc για την προσθήκη των αποστάσεων
        result_dfa = pd.concat([result_dfa, test_data])

    result_df = result_dfa

    #print(result_df)

    selected_columns = result_df[['file', 'Distance']]

    # Αποθήκευση των επιλεγμένων στηλών σε ένα αρχείο JSON
    json_file_name = output_file + ".json"

    selected_columns.to_json(json_file_name, orient='records', lines=True)

    # Δημιουργία του scatter plot για τα "yes"
    yes_trace = go.Scatter3d(x=result_df[result_df['REFACTORED'] == 1][param1],
                             y=result_df[result_df['REFACTORED'] == 1][param2],
                             z=result_df[result_df['REFACTORED'] == 1][param3],
                             mode='markers',
                             marker=dict(color="blue", size=10, opacity=1),
                             name="Yes")

    # Δημιουργία του scatter plot για τα "no"
    no_trace = go.Scatter3d(x=result_df[result_df['REFACTORED'] == 0][param1],
                            y=result_df[result_df['REFACTORED'] == 0][param2],
                            z=result_df[result_df['REFACTORED'] == 0][param3],
                            mode='markers',
                            marker=dict(color="black", size=10, opacity=1),
                            name="No")

    # Δημιουργία του scatter plot για το επιλεγμένο σημείο
    selected_trace = go.Scatter3d(x=data.loc[selected_row_indices][param1],
                                  y=data.loc[selected_row_indices][param2],
                                  z=data.loc[selected_row_indices][param3],
                                  mode='markers',
                                  marker=dict(color="red", size=15, opacity=1),
                                  name="Selected Class")

    # Κατασκευή του συνολικού διαγράμματος
    fig = go.Figure(data=[yes_trace, no_trace, selected_trace])

    # Προσαρμογή του layout
    fig.update_layout(scene=dict(xaxis_title=param1, yaxis_title=param2, zaxis_title=param3))

    name = output_file + "_plot.html"
    # Αποθήκευση του plot ως HTML
    fig.write_html(name)


if __name__ == '__main__':
    main()
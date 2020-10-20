"""
File name:     a1_code.py
Authors:       Maximilien Fathi
Date:          October 19, 2020
Description:   Code used to compare performances of different machine learning models
               using different datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

###############################################################################

# Global variables.

info_labels_1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
              "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]
    
info_labels_2 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

GNB_classifier = None
GNB_prediction = None
base_DT_classifier = None
base_DT_prediction = None
best_DT_estimator = None
best_DT_gridsearch = None
best_DT_prediction = None
PER_classifier = None
PER_prediction = None
base_MLP_classifier = None
base_MLP_prediction = None
best_MLP_estimator = None
best_MLP_gridsearch = None
best_MLP_prediction = None

###############################################################################

# Plots distribution of classes given a dataset file.
def plot_distributions(dataset_file, dataset_type, dataset_number):
    dataset_data = pd.read_csv(dataset_file, header=None)
    dataset_classes = dataset_data.iloc[:, -1].values
    # print(dataset_classes)
    
    frequency_distribution = Counter(dataset_classes)
    # print(frequency_distribution)
    plt.figure(figsize=(8, 5))
    plt.bar(*zip(*frequency_distribution.items()))
    plt.title(f'Distribution of number of instances in each class for {dataset_type}_{dataset_number} dataset')
    plt.xlabel("Classes")
    plt.ylabel("Number of instances")
    plt.xticks(range(len(frequency_distribution)))
    plt.savefig(f'Distribution_Plots/{dataset_type}_{dataset_number}_Distribution_Plot.png')
    plt.show()

###############################################################################

# Trains various ML models using training dataset files.
def train_models(train_file):
    global GNB_classifier, base_DT_classifier, best_DT_gridsearch, best_DT_estimator
    global PER_classifier, base_MLP_classifier, best_MLP_gridsearch, best_MLP_estimator
    
    train_data = pd.read_csv(train_file, header=None)
    train_values = train_data.iloc[:, 0:-1].values
    train_classes = train_data.iloc[:, -1].values
    
    # For Gaussian Naive Bayes model
    GNB_classifier = GaussianNB()
    GNB_classifier.fit(train_values, train_classes)
    
    # For base decision tree model
    base_DT_classifier = DecisionTreeClassifier(criterion='entropy')
    base_DT_classifier.fit(train_values, train_classes)
    
    # For best decision tree model
    best_DT_classifier = DecisionTreeClassifier()
    best_DT_parameters = {'criterion':['gini', 'entropy'], 'max_depth':[10, None],
                          'min_samples_split':[2, 3, 5, 10, 50],
                          'min_impurity_decrease':[0.0, 0.00005, 0.0005, 0.005, 0.05, 0.5, 5.0],
                          'class_weight':[None, 'balanced']}
    best_DT_gridsearch = GridSearchCV(best_DT_classifier, best_DT_parameters, n_jobs=-1)
    best_DT_gridsearch.fit(train_values, train_classes)
    best_DT_estimator = best_DT_gridsearch.best_estimator_
    print(f"best_DT_gridsearch.best_params_ is {best_DT_gridsearch.best_params_}")
    
    # For perceptron model
    PER_classifier = Perceptron()
    PER_classifier.fit(train_values, train_classes)
    
    # For base multi-layered perceptron model
    # Default max_iter of 200 gives convergence warnings as well as poor performance for dataset 1
    # Better with max_iter=4000 but takes longer
    base_MLP_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
    base_MLP_classifier.fit(train_values, train_classes)
    
    # For best multi-layered perceptron model
    best_MLP_classifier = MLPClassifier()
    best_MLP_parameters = {'activation':['logistic', 'tanh', 'relu', 'identity'],
                           'hidden_layer_sizes':[(20, 10, 30), (200,)], 'solver':['adam', 'sgd']}
    best_MLP_gridsearch = GridSearchCV(best_MLP_classifier, best_MLP_parameters, n_jobs=-1)
    best_MLP_gridsearch.fit(train_values, train_classes)
    best_MLP_estimator = best_MLP_gridsearch.best_estimator_
    print(f"best_MLP_gridsearch.best_params_ is {best_MLP_gridsearch.best_params_}")
    
###############################################################################

# Validates best-DT and best-MLP models using validation dataset files.
def validate_models(validation_file, info_labels):
    global best_DT_estimator, best_MLP_estimator
    
    validation_data = pd.read_csv(validation_file, header=None)
    validation_values = validation_data.iloc[:, 0:-1].values
    validation_classes = validation_data.iloc[:, -1].values
    print("VALIDATION CORRECT RESULTS\n", validation_classes)
    
    # For best decision tree model
    val_best_DT_prediction = best_DT_estimator.predict(validation_values)
    print("BEST-DT VALIDATION PREDICTED RESULTS\n", val_best_DT_prediction)
    print(classification_report(validation_classes, val_best_DT_prediction, target_names=info_labels))
    
    # For best multi-layered perceptron model
    val_best_MLP_prediction = best_MLP_estimator.predict(validation_values)
    print("BEST-MLP VALIDATION PREDICTED RESULTS\n", val_best_MLP_prediction)    
    print(classification_report(validation_classes, val_best_MLP_prediction, target_names=info_labels))

###############################################################################

# Predicts output for each ML model using feature values from labeled test files.
# Professor told us we could use labeled test file and disregard unlabeled test file.
def test_data(labeled_test_file, dataset_number):
    global GNB_classifier, GNB_prediction, base_DT_classifier, base_DT_prediction
    global best_DT_estimator, best_DT_prediction, PER_classifier, PER_prediction
    global base_MLP_classifier, base_MLP_prediction, best_DT_estimator, best_MLP_prediction
    
    # Fetching feature values from a given labeled test file
    labeled_test_data = pd.read_csv(labeled_test_file, header=None)
    test_values = labeled_test_data.iloc[:, 0:-1].values
    # Previous code => unlabeled_test_data = pd.read_csv(unlabeled_test_file, header=None)
    
    # For Gaussian Naive Bayes model
    model = "GNB"
    GNB_prediction = GNB_classifier.predict(test_values)
    print("GNB PREDICTED RESULTS\n", GNB_prediction)
    GNB_prediction_dataframe = pd.DataFrame(GNB_prediction)
    GNB_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)
    
    # For base decision tree model
    model = "Base-DT"
    base_DT_prediction = base_DT_classifier.predict(test_values)
    print("BASE-DT PREDICTED RESULTS\n", base_DT_prediction)
    base_DT_prediction_dataframe = pd.DataFrame(base_DT_prediction)
    base_DT_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)
    
    # For best decision tree model
    model = "Best-DT"
    best_DT_prediction = best_DT_estimator.predict(test_values)
    print("BEST-DT PREDICTED RESULTS\n", best_DT_prediction)
    best_DT_prediction_dataframe = pd.DataFrame(best_DT_prediction)
    best_DT_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)
    
    # For perceptron model
    model = "PER"
    PER_prediction = PER_classifier.predict(test_values)
    print("PERCEPTRON PREDICTED RESULTS\n", PER_prediction)
    PER_prediction_dataframe = pd.DataFrame(PER_prediction)
    PER_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)
    
    # For base multi-layered perceptron model
    model = "Base-MLP"
    base_MLP_prediction = base_MLP_classifier.predict(test_values)
    print("BASE-MPL PREDICTED RESULTS\n", base_MLP_prediction)
    base_MLP_prediction_dataframe = pd.DataFrame(base_MLP_prediction)
    base_MLP_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)
    
    # For best multi-layered perceptron model
    model = "Best-MLP"
    best_MLP_prediction = best_MLP_estimator.predict(test_values)
    print("BEST-MPL PREDICTED RESULTS\n", best_MLP_prediction)
    best_MLP_prediction_dataframe = pd.DataFrame(best_MLP_prediction)
    best_MLP_prediction_dataframe.to_csv(f'Output_Files/{model}-DS{dataset_number}.csv', header=False)

###############################################################################

# Plots a confusion matrix for each model and determines various performance metrics.
def evaluate_performance(labeled_test_file, info_labels, dataset_number):
    global GNB_prediction, base_DT_prediction, best_DT_prediction
    global PER_prediction, base_MLP_prediction, best_MLP_prediction

    labeled_test_data = pd.read_csv(labeled_test_file, header=None)
    correct_test_labels = labeled_test_data.iloc[:, -1].values
    print("CORRECT RESULTS\n", correct_test_labels)
    
    # For Gaussian Naive Bayes model
    model = "GNB"
    GNB_confusion_matrix = confusion_matrix(correct_test_labels, GNB_prediction)
    GNB_classification_report = classification_report(correct_test_labels, GNB_prediction, target_names=info_labels)
    export_confusion_matrix(GNB_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', GNB_classification_report, model)
    
    # For base decision tree model
    model = "Base-DT"
    base_DT_confusion_matrix = confusion_matrix(correct_test_labels, base_DT_prediction)
    base_DT_classification_report = classification_report(correct_test_labels, base_DT_prediction, target_names=info_labels)
    export_confusion_matrix(base_DT_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', base_DT_classification_report, model)
    
    # For best decision tree model
    model = "Best-DT"
    best_DT_confusion_matrix = confusion_matrix(correct_test_labels, best_DT_prediction)
    best_DT_classification_report = classification_report(correct_test_labels, best_DT_prediction, target_names=info_labels)
    export_confusion_matrix(best_DT_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', best_DT_classification_report, model)
    
    # For perceptron model
    model = "PER"
    PER_confusion_matrix = confusion_matrix(correct_test_labels, PER_prediction)
    PER_classification_report = classification_report(correct_test_labels, PER_prediction, target_names=info_labels)
    export_confusion_matrix(PER_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', PER_classification_report, model)
    
    # For base multi-layered perceptron model
    model = "Base-MLP"
    base_MLP_confusion_matrix = confusion_matrix(correct_test_labels, base_MLP_prediction)
    base_MLP_classification_report = classification_report(correct_test_labels, base_MLP_prediction, target_names=info_labels)
    export_confusion_matrix(base_MLP_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', base_MLP_classification_report, model)
    
    # For best multi-layered perceptron model
    model = "Best-MLP"
    best_MLP_confusion_matrix = confusion_matrix(correct_test_labels, best_MLP_prediction)
    best_MLP_classification_report = classification_report(correct_test_labels, best_MLP_prediction, target_names=info_labels)
    export_confusion_matrix(best_MLP_confusion_matrix, model, dataset_number)
    export_metrics(f'Output_Files/{model}-DS{dataset_number}_Performances.txt', best_MLP_classification_report, model)
        
###############################################################################

# Creates .PNG file showing the confusion matrix for a model.
def export_confusion_matrix(confusion_matrix, model, dataset_number):
    plt.figure(figsize=(8, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.title(f'Confusion matrix for {model}-DS{dataset_number}')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f'Output_Files/{model}-DS{dataset_number}_Confusion_matrix.png')
    plt.show()
    
###############################################################################

# Creates a .TXT file showing metrics for a model.
def export_metrics(output_file, classification_report, model):
    global best_DT_gridsearch, best_MLP_gridsearch
    with open(output_file, 'w') as file_object:
        file_object.write(classification_report)
        if model == "Best-DT":
            file_object.write(f'\nBest parameters are:\n {best_DT_gridsearch.best_params_}')
        elif model == "Best-MLP":
            file_object.write(f'\nBest parameters are:\n {best_MLP_gridsearch.best_params_}')

###############################################################################

# Runs the whole program for different datasets. 
def main():
    # Not mentioned which file to use for the plotting so I used all provided datasets.
    plot_distributions('Assig1-Dataset/train_1.csv', 'train', 1)
    plot_distributions('Assig1-Dataset/train_2.csv', 'train', 2)
    plot_distributions('Assig1-Dataset/val_1.csv', 'val', 1)
    plot_distributions('Assig1-Dataset/val_2.csv', 'val', 2)
    plot_distributions('Assig1-Dataset/test_with_label_1.csv', 'test', 1)
    plot_distributions('Assig1-Dataset/test_with_label_2.csv', 'test', 2)
    
    #--------------------------------------------------------------------------
    
    # Working with dataset 1
    train_models('Assig1-Dataset/train_1.csv')
    validate_models('Assig1-Dataset/val_1.csv', info_labels_1)
    test_data('Assig1-Dataset/test_with_label_1.csv', 1) # Previous => 'Assig1-Dataset/test_no_label_1.csv'
    evaluate_performance('Assig1-Dataset/test_with_label_1.csv', info_labels_1, 1)
    
    #--------------------------------------------------------------------------
    
    # Working with dataset 2
    train_models('Assig1-Dataset/train_2.csv')
    validate_models('Assig1-Dataset/val_2.csv', info_labels_2)
    test_data('Assig1-Dataset/test_with_label_2.csv', 2) # Previous => 'Assig1-Dataset/test_no_label_2.csv'
    evaluate_performance('Assig1-Dataset/test_with_label_2.csv', info_labels_2, 2)
    
###############################################################################
    
if __name__ == "__main__":
    main()

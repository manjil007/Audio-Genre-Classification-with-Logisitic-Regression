# Logistic Regression Music Genre Classification

This repository contains a machine learning project aimed at classifying music genres using a custom implementation of the Logistic Regression algorithm. The project is structured into two main components: feature extraction and model training & evaluation.

## Getting Started

### Prerequisites
Ensure you have Python installed along with the following libraries:
- NumPy
- pandas
- scikit-learn
- librosa
- matplotlib (optional, for plotting loss during training)

### Feature Extraction
Before training the model, features must be extracted from audio files. This is done using the `data_processor.py` script. To execute feature extraction, modify the file paths to your training and testing datasets within the script as shown below, then run the script.

Modify these lines in data_processor.py to point to your data directories
process_dataset_for_training('path/to/your/training_data')
process_dataset_for_testing('path/to/your/testing_data')


### Model Training and Evaluation
Once the features are extracted and saved as CSV files, the model can be trained using the train.py script. Before running the script, make sure the paths to the pre-extracted feature CSV files are correctly set. 

Modify these lines in train.py to use your feature CSV files
train_df = pd.read_csv('path/to/your/train_features.csv')
test_df = pd.read_csv('path/to/your/test_features.csv')

The script will train the Logistic Regression model, evaluate its performance, and save the predictions to a CSV file.

### LogisticRegression Class
The core of the project is the LogisticRegression class, which encapsulates the logistic regression algorithm. It offers functionality to fit the model to training data, make predictions on new data, and evaluate the model's accuracy.

### Running the Scripts and Description

1. **Feature Extraction**: The python script 'data_processor.py' contains all functions necessary for the feature extraction from audio files and to creae a CSV file that will be used by our model.

2. **Training Model**: The python script 'train.py' contains all functions necessary for the training of our model and generate predictions. Also prints out the cross validation accuracy as well as create a prediction file which can be submitted in the Kaggle to check accuracy. It calls 'LogisticRegression.py' for the model.

3. **Logistic Regression**: The python script 'LogisticRegression.py' contains the algorithm for logistic regression. It is called by 'train.py'

4. **Other Models**: The python script 'othermodels.py' contains the implementation of the other models - Random Forest, Gaussian Naive Bayes, Gradient Boosting Machines, and Support Vector Machines (SVM).

### Highest Kaggle Scores
Accuracy: 65% Leaderboard Position: 9 Date Submitted : 04/08/2024

Accuracy: 64% Date Submitted : 04/08/2024

### Contributions
Manjil - Wrote the code for the gradient descent function and also worked on finding the best combination of features that would yield the highest level of accuracy. Worked on the train.py file implementing the PCA, model training, prediction code, and evaluation part of the code. Worked on the report and contributed the Logistic Regression and Performance and Evaluation part of the report. 

Vincent - Worked on the Logistic Regression code by applying Manjil’s code on gradient descent and softmax. Functions include fit, predict, and evaluate. Researched appropriate features for genre classification and worked on Feature Extraction both in the code and in the report. Worked on expanding the dataset by trying multiple methods but was later discarded due to reduction in accuracy. 

Erick - Conducted research on spectral features and their potential relevance in genre classification. Worked on an early version of extracting these features which was later integrated into the completed data_processing file. Briefly worked on implementing KFold random_state, however, as it did not render accuracy improvements it was discarded. Worked on gradient descent, PCA, optimizations, and standardization in the report.

Abhinav - Researched librosa techniques and worked on processing data and feature extraction. Worked on sectioning of the songs and recombining sections from the same genre to see if they produced better results. For experiments, applied different sampling rates and hop rates to find the most accurate method. In the report, worked on the introduction, extracting information, results, and conclusion.

## Acknowledgments
Thanks to Professor. Estrada, and Teaching Assistants for their support during the project. Thanks to the our classmates and the machine learning community at large for the inspiration and support to develop this project.

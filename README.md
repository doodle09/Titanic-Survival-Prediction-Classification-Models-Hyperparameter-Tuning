Titanic Survival Prediction – Classification Models & Hyperparameter Tuning

This project builds a complete machine learning workflow to predict passenger survival on the Titanic using classification algorithms. The goal is to compare different models, tune them for better accuracy, and understand which factors influence survival the most.
The analysis includes data cleaning, preprocessing, feature encoding, model training, hyperparameter tuning, and performance comparison using multiple evaluation metrics.

The models implemented include Logistic Regression, Random Forest, and Support Vector Machine (SVM), followed by GridSearchCV and RandomizedSearchCV to find optimal parameters.

Project Workflow
1. Data Loading & Missing Value Handling

The Titanic dataset is loaded directly from Seaborn.
Key cleaning steps include:

Removing irrelevant columns (like deck, alive, adult_male, etc.)

Dropping rows missing critical fields like embarked or fare

Filling age with the median

Cleaning all remaining missing values to prepare the data for modeling

2. Encoding & Scaling

Categorical features (sex, embarked) are converted into numerical values.
Feature scaling is applied using StandardScaler, ensuring models like SVM and Logistic Regression perform correctly.

3. Model Training – Base Models

Three baseline models are trained and evaluated:

Logistic Regression

Random Forest Classifier

SVM (Support Vector Machine)

Each model is evaluated on:

Accuracy

Precision

Recall

F1 Score

This helps compare performance beyond accuracy alone.

4. Hyperparameter Tuning

To improve performance:

Random Forest – GridSearchCV

Parameters tuned:

Number of trees (n_estimators)

Maximum depth (max_depth)

SVM – RandomizedSearchCV

Randomized search explores combinations of:

C (regularization strength)

Kernel type

Gamma settings

Both tuned models are evaluated using classification reports to show precision, recall, and F1 across both classes.

5. Best Model Selection

After tuning, the accuracy of both optimized models is compared:

Tuned Random Forest

Tuned SVM

The model with the highest accuracy on the test set is selected as the final best performer.

6. Visualization

A bar chart is created to compare baseline model performances side-by-side for accuracy, precision, recall, and F1 score.
This visualization helps clearly show how each model performs before tuning.

Conclusion

This project demonstrates the full pipeline of building and optimizing classification models:

Data cleaning and preprocessing

Encoding and scaling

Training multiple machine learning models

Hyperparameter tuning using GridSearchCV & RandomizedSearchCV

Comparing model performance and selecting the best algorithm

The final result highlights how tuning significantly improves performance and gives insights into what drives survival on the Titanic.

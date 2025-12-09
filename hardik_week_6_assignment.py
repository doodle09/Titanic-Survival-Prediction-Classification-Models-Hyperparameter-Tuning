import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, classification_report

### loading titanic dataset from seaborn

df=sns.load_dataset("titanic")

### first 5 rows

print("\ndata preview:")
print(df.head())

### checking missing values

missing_percent= df.isnull().mean()*100
missing_percent= missing_percent[missing_percent>0].sort_values(ascending=False)
print("Missing Values (%):\n")
print(missing_percent)

### droping irrevelent columns,rows with missing 'embarked' or 'fare'

df.drop(columns=['who', 'deck', 'embark_town', 'alive', 'class', 'adult_male'], inplace=True)
df.dropna(subset=['embarked', 'fare'], inplace=True)

### filling missing age with median

df['age'].fillna(df['age'].median(), inplace=True)

### encode categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

### droping remaning missing values
df.dropna(inplace=True)

### test,train
X = df.drop(columns='survived')
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

### traning models

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

### display results

print("model performance (base models):")
for model, metrics in results.items():
    print(f"\n{model}")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

### Hyperparameter tuning - Random Forest with GridSearchCV
# this took forever to understand
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print("\nBest Random Forest (GridSearchCV):")
print(grid_rf.best_params_)

### Hyperparameter Tuning - SVM with RandomizedSearchCV

param_dist_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

random_svm = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=5, cv=3, scoring='accuracy', random_state=42)
random_svm.fit(X_train, y_train)
best_svm = random_svm.best_estimator_

print("\nBest SVM (RandomizedSearchCV):")
print(random_svm.best_params_)

# Final evaluation of tuned models
print("\nEvaluation of Tuned Models:")
for model_name, model in [("Tuned Random Forest", best_rf), ("Tuned SVM", best_svm)]:
    y_pred = model.predict(X_test)
    print(f"\n{model_name}")
    print(classification_report(y_test, y_pred))

### compare tuned vs base models

tuned_rf_acc = accuracy_score(y_test, best_rf.predict(X_test))
tuned_svm_acc = accuracy_score(y_test, best_svm.predict(X_test))
print(f"\ntuned random forest accuracy: {tuned_rf_acc:.4f}")  
print(f"tuned svm accuracy: {tuned_svm_acc:.4f}")

### best model selection

if tuned_rf_acc > tuned_svm_acc:
    print(f"\nbest performing model: tuned random forest ({tuned_rf_acc:.4f})")
else:
    print(f"\nbest performing model: tuned svm ({tuned_svm_acc:.4f})")


df_results = pd.DataFrame(results).T  

### graph plotting

sns.set_style("whitegrid")
df_results.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Model Comparison - Base Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
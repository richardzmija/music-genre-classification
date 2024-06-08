import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

SEED = 14
TEST_RATIO = 0.2
file_path = r"data\features_3_sec.csv"
model_name = "xgb_model"
encoder_name = "xgb_encoder"

# Data loading
data = pd.read_csv(file_path)

# Remove 'filename' and 'length' columns as they are not
# needed for classification
data = data.drop(columns=["filename", "length"])

# Split the data into features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_RATIO,
                                                    random_state=SEED)

# Create and initialize a Gradient Boosting Machine classifier
# from XGBoost that uses C++ as backend for faster computation
DECISION_TREES = 1000
LEARNING_RATE = 0.1
MAX_DEPTH = 5

xgb_classifier = xgb.XGBClassifier(n_estimators=DECISION_TREES,
                                   learning_rate=LEARNING_RATE,
                                   max_depth=MAX_DEPTH,
                                   random_state=SEED)

# Train the model
xgb_classifier.fit(X_train, y_train)

# Save the model
model_filename = model_name + ".pkl"
encoder_filename = encoder_name + ".pkl"

joblib.dump(xgb_classifier, model_filename)
joblib.dump(label_encoder, encoder_filename)

# Test and evaluate the model
y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"XGBoost Classification Report:\n{report}")

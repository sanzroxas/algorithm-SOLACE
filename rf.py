import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight #For handling imbalance class
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.utils import to_categorical  # Corrected import


# --- Data Loading ---
df = pd.read_csv("dataset/dataset.csv")
symptom_weights = pd.read_csv("dataset/Symptom-severity.csv")
symptom_precautions = pd.read_csv("dataset/symptom_precaution.csv")  # Load precautions
symptom_descriptions = pd.read_csv("dataset/symptom_Description.csv") # Load descriptions


# --- Data Preprocessing ---
# Handle Missing Values (replace with empty string for now, can be improved)
df = df.fillna('')

# Create a dictionary for symptom weights
symptom_weights_dict = dict(zip(symptom_weights['Symptom'], symptom_weights['weight']))

# Function to create weighted symptom columns, modified for time series
def create_weighted_symptom_cols(row):
    weighted_symptoms = {}
    for col in df.columns[1:]:  # Iterate through symptom columns
        symptom = row[col].strip()  # Get symptom name and remove whitespace
        if symptom:  # Check if the symptom is not empty
            if symptom in symptom_weights_dict:
                weighted_symptoms[f"weighted_{symptom}"] = symptom_weights_dict[symptom]
            else:
                 # Handle unseen symptoms (assign a default weight or skip)
                weighted_symptoms[f"weighted_{symptom}"] = 1 # Assigning default value of 1
    return pd.Series(weighted_symptoms)  # Return as a pandas series for faster operation

# Apply the function to each row to create new columns
weighted_cols = df.apply(create_weighted_symptom_cols, axis=1)
# Concatenate with the original dataframe
df = pd.concat([df, weighted_cols], axis=1)
# Drop the original symptom columns
df = df.drop(columns=df.columns[1:18]) # Drop original symptom columns (Symptom_1 to Symptom_17)
df = df.fillna(0) # Replace NaN with 0


# 3. One-Hot Encode the 'Disease' column (Target Variable)
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])
# Store the mapping for later use
disease_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(disease_mapping)

# --- Synthetic Time Series Data Generation (More Realistic) ---

def generate_synthetic_time_series_data(df, min_timesteps=3, max_timesteps=7, random_seed=42):
    np.random.seed(random_seed)
    all_data = []
    all_labels = []

    # Get the feature names *before* the loop.  CRITICAL for consistent ordering.
    feature_names = df.columns[1:].tolist()  # List of weighted symptom names

    for _, row in df.iterrows():
        disease = row['Disease']
        timesteps = np.random.randint(min_timesteps, max_timesteps + 1)  # Variable timesteps
        patient_data = []
        initial_symptoms = row[1:].values.astype(float)

        for t in range(timesteps):
            current_symptoms = initial_symptoms.copy()

            for i in range(len(current_symptoms)):
                if current_symptoms[i] > 0:
                    # More complex temporal changes + correlation
                    change_type = np.random.choice(["increase", "decrease", "fluctuate", "no_change"],
                                                   p=[0.3, 0.2, 0.3, 0.2]) #p for probability

                    if change_type == "increase":
                        change = np.random.normal(0.5, 0.5)  # Larger increase
                    elif change_type == "decrease":
                        change = np.random.normal(-0.5, 0.5) # Decrease
                    elif change_type == "fluctuate":
                        change = np.random.normal(0, 1.0)   # Larger fluctuation
                    else:
                        change = 0

                    current_symptoms[i] += change
                    current_symptoms[i] = max(0, current_symptoms[i])  # Keep non-negative

                    # Example of correlated symptom (adjust as needed)
                    if "weighted_cough" in feature_names: #check first if exists
                        cough_index = feature_names.index("weighted_cough")
                        if feature_names[i] == "weighted_breathlessness" and current_symptoms[cough_index] > 3:
                            current_symptoms[i] += 0.3 #breathlessness increases with cough



            #Symptom Onset
            if t > 0:  # After the first time step
                for i in range(len(current_symptoms)):
                     if current_symptoms[i] == 0:  # If symptom was NOT present
                        if np.random.rand() < 0.05: # 5% chance of NEW symptom appearing
                            # Find corresponding symptom without "weighted"
                            symptom_name = feature_names[i].replace("weighted_", "")
                            # print(symptom_name)
                            if symptom_name in symptom_weights_dict:
                                current_symptoms[i] = symptom_weights_dict[symptom_name] # Set to its initial weight
                            else:
                                current_symptoms[i] = 1


            # Introduce some missing data (NaN values)
            for i in range(len(current_symptoms)):
                if np.random.rand() < 0.2:  # 20% chance of a symptom being missing at a timestep
                    current_symptoms[i] = 0.0 # Use 0 for masking

            patient_data.append(current_symptoms)

        all_data.append(patient_data)
        all_labels.append(disease)

    return np.array(all_data, dtype=object), np.array(all_labels)


# Generate data
timesteps = 5  # Number of time steps per patient
X, y = generate_synthetic_time_series_data(df)

# --- Padding Sequences to a Uniform Length ---
# Find the maximum sequence length
max_length = max(len(seq) for seq in X)

# Pad sequences using a loop
X_padded = []
for seq in X:
    # Convert the sequence to a NumPy array
    seq_array = np.array(seq)
    # Pad the sequence with zeros at the end
    padding_length = max_length - len(seq)
    padded_seq = np.pad(seq_array, ((0, padding_length), (0, 0)), mode='constant')
    X_padded.append(padded_seq)

# Convert list to a NumPy array
X = np.array(X_padded)

# --- Data Scaling ---
# Scale the data.  Important for LSTMs.
scaler = MinMaxScaler()
# Reshape to 2D for scaling, then reshape back to 3D
original_shape = X.shape
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1]))  # Reshape and scale
X_scaled = X_scaled.reshape(original_shape)  # Put back into time-series format


# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test_cat = to_categorical(y_test, num_classes=len(label_encoder.classes_))



# --- Random Forest Model ---

# Use only the LAST time step for the Random Forest
X_train_rf = X_train[:, -1, :]  # (samples, features) - last timestep
X_test_rf = X_test[:, -1, :]    # (samples, features) - last timestep

param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],  # Added more options
    'max_depth': [5, 8, 10, 12, None],    # Added more options
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_model = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(rf_model, param_grid_rf, n_iter=20, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_rf.fit(X_train_rf, y_train)  # y_train, not y_train_cat
best_rf_model = random_search_rf.best_estimator_
print("Best Random Forest Parameters:", random_search_rf.best_params_)

y_pred_rf = best_rf_model.predict(X_test_rf)

print("--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.show()

importances = best_rf_model.feature_importances_
# Feature names are now directly the symptom names!
feature_importance_df = pd.DataFrame({'Feature': df.columns[1:], 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances (Random Forest):\n", feature_importance_df.head(20))

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Random Forest Feature Importance (Top 20)')
plt.tight_layout()
plt.show()


# --- LSTM Model Definition ---

lstm_model = Sequential()
lstm_model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(LSTM(32, return_sequences=False,
                   dropout=0.3, recurrent_dropout=0.3,
                   kernel_regularizer=regularizers.l2(0.02),
                   recurrent_regularizer=regularizers.l2(0.02),
                   bias_regularizer=regularizers.l2(0.02)))
lstm_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.summary()


# --- LSTM Model Training with Early Stopping ---

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(X_train, y_train_cat, epochs=100, batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping])



# --- LSTM Model Definition ---

lstm_model = Sequential()
lstm_model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2]))) #Timesteps, Features
lstm_model.add(LSTM(32, return_sequences=False,  # Reduced LSTM units
                   dropout=0.3, recurrent_dropout=0.3,  # Increased dropout
                   kernel_regularizer=regularizers.l2(0.02), # Increased L2
                   recurrent_regularizer=regularizers.l2(0.02),
                   bias_regularizer=regularizers.l2(0.02)))
lstm_model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.summary()


# --- LSTM Model Training with Early Stopping ---

# Define Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the LSTM Model
history = lstm_model.fit(X_train, y_train_cat, epochs=100, batch_size=32,  # Increased epochs
                    validation_split=0.2,  # Increased validation split
                    callbacks=[early_stopping] # Add early stopping
                    )




# --- LSTM Model Evaluation ---

# Evaluate the LSTM Model
loss, accuracy = lstm_model.evaluate(X_test, y_test_cat)
print(f"LSTM Test Accuracy: {accuracy}")

# Make predictions
y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)  # Convert probabilities to class labels
y_true_classes = np.argmax(y_test_cat, axis=1)

print("Classification Report (LSTM):\n", classification_report(y_true_classes, y_pred_lstm_classes, target_names=label_encoder.classes_))


# --- Visualization ---
# Confusion Matrix Visualization
'''
cm_lstm = confusion_matrix(y_true_classes, y_pred_lstm_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_) # Using the stored mapping
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - LSTM')
plt.show()


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

# --- Post-Prediction: Displaying Precautions and Descriptions ---

# Function to get precautions and description for a predicted disease
def get_disease_info(predicted_disease_label):
    predicted_disease_name = label_encoder.inverse_transform([predicted_disease_label])[0]

    # Get Precautions
    precaution_row = symptom_precautions[symptom_precautions['Disease'] == predicted_disease_name]
    if not precaution_row.empty:
        precautions = precaution_row.iloc[0, 1:].values.tolist()  # Get all precaution columns
        precautions = [p for p in precautions if isinstance(p, str) and p.strip()]  # Clean up
    else:
        precautions = ["No precautions found."]

    # Get Description
    description_row = symptom_descriptions[symptom_descriptions['Disease'] == predicted_disease_name]
    if not description_row.empty:
        description = description_row['Description'].iloc[0]
    else:
        description = "No description found."

    return predicted_disease_name, precautions, description


# Example Usage (after making predictions with either model):
# Assuming we just made predictions with the LSTM model:
for i in range(5):  # Show info for the first 5 predictions
    predicted_label = y_pred_lstm_classes[i]
    disease_name, precautions, description = get_disease_info(predicted_label)

    print(f"\n--- Prediction {i+1} ---")
    print(f"Predicted Disease (Label): {predicted_label}")
    print(f"Predicted Disease (Name): {disease_name}")
    print("Precautions:")
    for j, p in enumerate(precautions):
        print(f"  {j+1}. {p}")
    print(f"Description: {description}")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
import csv


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    #features = ['lactate_max', 'ph_min', 'platelets_min', 'inr_max', 'calcium_min', 'ptt_max']
    features = ['lactate_max', 'ph_min', 'platelets_min', 'inr_max',
            'calcium_min', 'ptt_max',
            'creat_low_past_7day', 'creat_low_past_48hr', 'creat',
            'creatinine_min', 'creatinine_max',
            'bicarbonate_min', 'bicarbonate_max', 'baseexcess_min',
            'baseexcess_max', 'chloride_min', 'chloride_max',
            'potassium_min', 'potassium_max', 'sodium_min', 'sodium_max',
            'alt_min', 'alt_max', 'ast_min', 'ast_max',
            'inr_min', 'pt_min', 'pt_max', 'ptt_min',
            'hemoglobin_min', 'hemoglobin_max', 'hematocrit_min',
            'hematocrit_max', 'platelets_max', 'uo_rt_6hr', 'uo_rt_12hr',
            'uo_rt_24hr', 'aki_stage_uo', 'admission_age', 'los_hospital',
            'los_icu', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean',
            'sbp_min', 'sbp_max', 'sbp_mean', 'dbp_min', 'dbp_max',
            'dbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean',
            'spo2_min', 'spo2_max', 'spo2_mean']

    df = df[features]

    # print(df.dtypes)

    # Add binary target for citric acid overdose detection
    def detect_citric_acid_overdose(row):
        if row['lactate_max'] > 2.5 and row['ph_min'] < 7.35 and row['platelets_min'] < 150 and row['inr_max'] > 1.5:
            return 1  # Overdose detected
        else:
            return 0  # No overdose

    # Add continuous target for anticoagulation adjustment (only if overdose detected)
    def adjust_anticoagulation(row):
        if row['citric_acid_overdose'] == 1:  # Only adjust if overdose detected
            adjustment = 0.5  # Base adjustment value
            if row['calcium_min'] < 0.8:
                adjustment += 0.5
            elif row['calcium_min'] < 1.0:
                adjustment += 0.3
            if row['ptt_max'] > 80:
                adjustment += 0.3
            elif row['ptt_max'] > 60:
                adjustment += 0.2
            if row['platelets_min'] < 100:
                adjustment -= 0.1
            if row['inr_max'] > 2.0:
                adjustment -= 0.3
            return adjustment
        else:
            return 0  # No adjustment if no overdose

    df['citric_acid_overdose'] = df.apply(detect_citric_acid_overdose, axis=1)
    df['anticoagulation_adjustment'] = df.apply(adjust_anticoagulation, axis=1)


    df.fillna(df.mean(), inplace=True)

    X = df[features].values
    y_binary = df['citric_acid_overdose'].values
    y_continuous = df['anticoagulation_adjustment'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply SMOTE for balancing the binary classification task
    smote = SMOTE()
    X_resampled, y_binary_resampled = smote.fit_resample(X, y_binary)

    # Now we need to match the continuous labels with the resampled binary labels
    resampled_indices = smote.fit_resample(np.arange(len(y_binary)).reshape(-1, 1), y_binary)[0].flatten()
    y_continuous_resampled = y_continuous[resampled_indices]

    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test = train_test_split(
        X_resampled, y_binary_resampled, y_continuous_resampled, test_size=0.2, random_state=42
    )

    # return X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test
    return X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler

def create_non_iid_data(X, y_binary, y_continuous, num_clients):
    # Separate data by binary label (overdose or not)
    overdose_indices = np.where(y_binary == 1)[0]
    non_overdose_indices = np.where(y_binary == 0)[0]

    # Shuffle the indices within each group
    np.random.shuffle(overdose_indices)
    np.random.shuffle(non_overdose_indices)

    # Split the overdose and non-overdose data into uneven parts
    overdose_splits = np.array_split(overdose_indices, num_clients)
    non_overdose_splits = np.array_split(non_overdose_indices, num_clients)

    X_splits = []
    y_binary_splits = []
    y_continuous_splits = []

    for i in range(num_clients):
        # Combine some overdose and non-overdose examples for each client
        client_indices = np.concatenate([overdose_splits[i], non_overdose_splits[i]])
        X_splits.append(X[client_indices])
        y_binary_splits.append(y_binary[client_indices])
        y_continuous_splits.append(y_continuous[client_indices])

    return X_splits, y_binary_splits, y_continuous_splits


# Create Binary Classification Model with Dropout
def create_binary_classification_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)  # Adding Dropout
    x = GRU(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)  # Adding Dropout
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    binary_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, binary_output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create Continuous Regression Model
def create_continuous_regression_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = GRU(64, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    continuous_output = Dense(1, activation='linear')(x)

    model = Model(inputs, continuous_output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

# Local training function for binary model
def local_training_binary(model, X_client, y_binary_client, epochs):
    history = model.fit(X_client, y_binary_client, epochs=epochs, batch_size=32, verbose=0)
    return model.get_weights(), history.history['accuracy']

# Local training function for continuous model
def local_training_continuous(model, X_client, y_continuous_client, epochs):
    history = model.fit(X_client, y_continuous_client, epochs=epochs, batch_size=32, verbose=0)
    return model.get_weights(), history.history['mean_absolute_error']


def federated_server(X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, communication_rounds, learning_rate, local_epochs, num_edge_devices, privacy_param):
    binary_model = create_binary_classification_model(input_shape=(X_train.shape[1], 1))
    continuous_model = create_continuous_regression_model(input_shape=(X_train.shape[1], 1))

    '''

    X_splits = np.array_split(X_train, num_edge_devices)
    y_binary_splits = np.array_split(y_binary_train, num_edge_devices)
    y_continuous_splits = np.array_split(y_continuous_train, num_edge_devices)

    '''

    # Use non-IID data splitting
    X_splits, y_binary_splits, y_continuous_splits = create_non_iid_data(X_train, y_binary_train, y_continuous_train, num_edge_devices)

    training_accuracy = []
    training_loss = []
    testing_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'latency': []}

    # Open a CSV file to store the testing metrics for each round
    with open('federated_testing_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Round', 'Binary Loss', 'Binary Accuracy', 'Continuous Loss', 'Continuous MAE', 'Sensitivity', 'Specificity', 'Latency'])

        for t in range(communication_rounds):
            selected_clients = select_clients_based_on_budget(num_edge_devices, X_splits, y_binary_splits, y_continuous_splits, budget_threshold=0.5)

            # Initialize models for each selected client
            client_binary_models = [create_binary_classification_model(input_shape=(X_train.shape[1], 1)) for _ in selected_clients]
            client_continuous_models = [create_continuous_regression_model(input_shape=(X_train.shape[1], 1)) for _ in selected_clients]

            binary_client_updates = []
            continuous_client_updates = []
            binary_accuracies = []
            continuous_losses = []

            for i, client_idx in enumerate(selected_clients):
                binary_model_local = create_binary_classification_model(input_shape=(X_train.shape[1], 1))
                continuous_model_local = create_continuous_regression_model(input_shape=(X_train.shape[1], 1))

                # Set initial global model weights for local training
                binary_model_local.set_weights(binary_model.get_weights())
                continuous_model_local.set_weights(continuous_model.get_weights())

                # Perform local training
                binary_weights, binary_acc = local_training_binary(binary_model_local, X_splits[client_idx], y_binary_splits[client_idx], local_epochs)
                continuous_weights, continuous_mae = local_training_continuous(continuous_model_local, X_splits[client_idx], y_continuous_splits[client_idx], local_epochs)

                # Add noise to weights for privacy
                noisy_binary_weights = [w + np.random.laplace(0, privacy_param, w.shape) for w in binary_weights]
                noisy_continuous_weights = [w + np.random.laplace(0, privacy_param, w.shape) for w in continuous_weights]

                # Store updates with noise
                binary_client_updates.append(noisy_binary_weights)
                continuous_client_updates.append(noisy_continuous_weights)

                binary_accuracies.extend(binary_acc)
                continuous_losses.extend(continuous_mae)

            # Aggregate the noisy model updates
            binary_global_weights = aggregate_weights(binary_client_updates)
            continuous_global_weights = aggregate_weights(continuous_client_updates)

            # Update global models with aggregated weights
            binary_model.set_weights(binary_global_weights)
            continuous_model.set_weights(continuous_global_weights)

            # Evaluate the global models on the test set
            binary_loss, binary_acc, binary_metrics = evaluate_global_model(binary_model, X_test, y_binary_test, 'binary', threshold=0.4)
            continuous_loss, continuous_mae, continuous_metrics = evaluate_global_model(continuous_model, X_test, y_continuous_test, 'continuous')

            training_accuracy.append(np.mean(binary_accuracies))
            training_loss.append(np.mean(continuous_losses))

            testing_metrics['accuracy'].append(binary_acc)
            testing_metrics['sensitivity'].append(binary_metrics['sensitivity'])
            testing_metrics['specificity'].append(binary_metrics['specificity'])
            testing_metrics['latency'].append(binary_metrics['latency'])

            # Write the current round's metrics to the CSV file
            writer.writerow([t, binary_loss, binary_acc, continuous_loss, continuous_mae, binary_metrics['sensitivity'], binary_metrics['specificity'], binary_metrics['latency']])

            print(f"Round {t}: Binary Loss = {binary_loss}, Binary Accuracy = {binary_acc}, Continuous Loss = {continuous_loss}, Continuous MAE = {continuous_mae}")

    # Plot metrics for visualization
    plot_metrics(training_accuracy, training_loss, testing_metrics)

    return binary_model, continuous_model


def aggregate_weights(client_updates):
    num_clients = len(client_updates)
    avg_weights = []
    for weights_list in zip(*client_updates):
        avg_weights.append(np.mean(weights_list, axis=0))
    return avg_weights

def get_resource_metrics(client_id):
    """
    Retrieve real resource metrics for a client.
    Replace this with actual data retrieval logic.
    """
    # Randomly generated
    cpu_usage = np.random.uniform(0.1, 1.0)
    memory_usage = np.random.uniform(0.1, 1.0)
    network_latency = np.random.uniform(0.1, 1.0)
    return cpu_usage, memory_usage, network_latency

def calculate_resource_cost(cpu_usage, memory_usage, network_latency):
    """
    Calculate the resource cost based on CPU, memory, and network metrics.
    Adjust the weights based on the importance of each metric.
    """
    cpu_weight = 0.4
    memory_weight = 0.4
    latency_weight = 0.2

    return (cpu_usage * cpu_weight) + (memory_usage * memory_weight) + (network_latency * latency_weight)

def select_clients_based_on_budget(num_clients, X_splits, y_binary_splits, y_continuous_splits, budget_threshold):
    """
    Select clients based on their resource cost and available budget.

    Args:
    - num_clients: Total number of clients.
    - X_splits: Data splits for each client.
    - y_binary_splits: Binary labels splits for each client.
    - y_continuous_splits: Continuous labels splits for each client.
    - budget_threshold: Maximum allowable resource cost based on the budget.

    Returns:
    - selected_clients: Array of client indices selected for the round.
    """
    budgets = [np.random.uniform(0.5, 1.5) for _ in range(num_clients)]  # budgets for each client
    selected_clients = []

    for client_id in range(num_clients):
        cpu_usage, memory_usage, network_latency = get_resource_metrics(client_id)
        resource_cost = calculate_resource_cost(cpu_usage, memory_usage, network_latency)
        available_budget = budgets[client_id]

        # Check if the resource cost is within budget and below the threshold
        if resource_cost <= available_budget and resource_cost <= budget_threshold:
            selected_clients.append(client_id)

    # Ensure that we select a subset based on the total number of clients and threshold
    num_selected = min(len(selected_clients), int(num_clients * budget_threshold))

    # If not enough clients meet the criteria, select as many as possible
    if len(selected_clients) < num_selected:
        num_selected = len(selected_clients)

    return np.random.choice(selected_clients, num_selected, replace=False)


def evaluate_global_model(model, X_test, y_test, task, threshold=0.6):
    start_time = time.time()
    predictions = model.predict(X_test)
    latency = time.time() - start_time

    if task == 'binary':
        binary_predictions = (predictions.flatten() > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        loss, _ = model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy, {'sensitivity': sensitivity, 'specificity': specificity, 'latency': latency}

    elif task == 'continuous':
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        return loss, mae, {'latency': latency}


def cloud_computing_model(X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test):
    # Binary Classification Model for Cloud Computing
    binary_model = create_binary_classification_model(input_shape=(X_train.shape[1], 1))
    continuous_model = create_continuous_regression_model(input_shape=(X_train.shape[1], 1))

    # Train Binary Classification Model
    print("Training Binary Classification Model in Cloud...")
    binary_history = binary_model.fit(
        X_train, y_binary_train,
        validation_data=(X_test, y_binary_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Train Continuous Regression Model
    print("Training Continuous Regression Model in Cloud...")
    continuous_history = continuous_model.fit(
        X_train, y_continuous_train,
        validation_data=(X_test, y_continuous_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Evaluate Binary Classification Model
    binary_loss, binary_accuracy, binary_metrics = evaluate_global_model(
        binary_model, X_test, y_binary_test, task="binary", threshold=0.5
    )

    # Evaluate Continuous Regression Model
    continuous_loss, continuous_mae, continuous_metrics = evaluate_global_model(
        continuous_model, X_test, y_continuous_test, task="continuous"
    )

    print("\nCloud Model Evaluation:")
    print(f"Binary Model - Loss: {binary_loss}, Accuracy: {binary_accuracy}")
    print(f"Continuous Model - Loss: {continuous_loss}, MAE: {continuous_mae}")

    # Plot Cloud Model Performance
    plt.figure(figsize=(10, 6))

    # Binary Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(binary_history.history['accuracy'], label="Train Accuracy")
    plt.plot(binary_history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Binary Classification Accuracy")
    plt.legend()

    # Continuous Loss
    plt.subplot(1, 2, 2)
    plt.plot(continuous_history.history['loss'], label="Train Loss")
    plt.plot(continuous_history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Continuous Regression Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nCloud Metrics:")
    print(f"Binary Model Metrics: {binary_metrics}")
    print(f"Continuous Model Metrics: {continuous_metrics}")

    return binary_metrics, continuous_metrics

def edge_computing_model(X_train_list, X_test_list, y_binary_train_list, y_binary_test_list,
                         y_continuous_train_list, y_continuous_test_list):
    # Initialize lists to store metrics from each edge device
    binary_metrics_list = []
    continuous_metrics_list = []

    print("Training and evaluating models on edge devices...")

    for i in range(len(X_train_list)):
        print(f"\nEdge Device {i+1}:")

        # Create models for each edge device
        binary_model = create_binary_classification_model(input_shape=(X_train_list[i].shape[1], 1))
        continuous_model = create_continuous_regression_model(input_shape=(X_train_list[i].shape[1], 1))

        # Train Binary Classification Model
        print(f"Training Binary Classification Model on Edge Device {i+1}...")
        binary_model.fit(
            X_train_list[i], y_binary_train_list[i],
            validation_data=(X_test_list[i], y_binary_test_list[i]),
            epochs=30,
            batch_size=16,
            verbose=1
        )

        # Train Continuous Regression Model
        print(f"Training Continuous Regression Model on Edge Device {i+1}...")
        continuous_model.fit(
            X_train_list[i], y_continuous_train_list[i],
            validation_data=(X_test_list[i], y_continuous_test_list[i]),
            epochs=30,
            batch_size=16,
            verbose=1
        )

        # Evaluate Binary Classification Model
        binary_loss, binary_accuracy, binary_metrics = evaluate_global_model(
            binary_model, X_test_list[i], y_binary_test_list[i], task="binary", threshold=0.5
        )

        # Evaluate Continuous Regression Model
        continuous_loss, continuous_mae, continuous_metrics = evaluate_global_model(
            continuous_model, X_test_list[i], y_continuous_test_list[i], task="continuous"
        )

        # Store Metrics
        binary_metrics_list.append(binary_metrics)
        continuous_metrics_list.append(continuous_metrics)

        print(f"Edge Device {i+1} Metrics:")
        print(f"Binary Model - Loss: {binary_loss}, Accuracy: {binary_accuracy}")
        print(f"Continuous Model - Loss: {continuous_loss}, MAE: {continuous_mae}")

    # Aggregate Metrics
    print("\nAggregating metrics across all edge devices...")
    aggregated_binary_metrics = aggregate_edge_metrics(binary_metrics_list)
    aggregated_continuous_metrics = aggregate_edge_metrics(continuous_metrics_list)

    print("\nAggregated Metrics:")
    print(f"Binary Model Metrics: {aggregated_binary_metrics}")
    print(f"Continuous Model Metrics: {aggregated_continuous_metrics}")

    return aggregated_binary_metrics, aggregated_continuous_metrics


# Helper function to aggregate metrics
def aggregate_edge_metrics(metrics_list):
    """Aggregate metrics across edge devices."""
    total_metrics = {key: 0 for key in metrics_list[0].keys()}
    for metrics in metrics_list:
        for key, value in metrics.items():
            total_metrics[key] += value
    # Take the average
    num_devices = len(metrics_list)
    return {key: value / num_devices for key, value in total_metrics.items()}


# Simulated Edge Data Splitting
def simulate_edge_data(X, y_binary, y_continuous, num_edges):
    """Split data into chunks to simulate edge devices."""
    chunk_size = len(X) // num_edges
    X_split = [X[i*chunk_size:(i+1)*chunk_size] for i in range(num_edges)]
    y_binary_split = [y_binary[i*chunk_size:(i+1)*chunk_size] for i in range(num_edges)]
    y_continuous_split = [y_continuous[i*chunk_size:(i+1)*chunk_size] for i in range(num_edges)]
    return X_split, y_binary_split, y_continuous_split

def plot_metrics(training_accuracy, training_loss, testing_metrics):
    # Plot training and testing metrics
    rounds = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(rounds, training_accuracy, label="Training Accuracy")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(rounds, training_loss, label="Training Loss")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(rounds, testing_metrics['accuracy'], label="Test Accuracy")
    plt.plot(rounds, testing_metrics['sensitivity'], label="Sensitivity")
    plt.plot(rounds, testing_metrics['specificity'], label="Specificity")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Metrics")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(rounds, testing_metrics['latency'], label="Latency")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Latency (seconds)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_combined_metrics(fed_training_accuracy, fed_training_loss, fed_testing_metrics,
                          edge_training_accuracy, edge_training_loss, edge_testing_metrics,
                          cloud_training_accuracy, cloud_training_loss, cloud_testing_metrics):
    rounds = range(1, len(fed_training_accuracy) + 1)

    plt.figure(figsize=(14, 10))

    # Training Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(rounds, fed_training_accuracy, label="Federated Edge", color='blue')
    plt.plot(rounds, edge_training_accuracy, label="Edge (Non-Federated)", color='orange')
    plt.plot(rounds, cloud_training_accuracy, label="Cloud-Based", color='green')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.legend()

    # Training Loss
    plt.subplot(2, 2, 2)
    plt.plot(rounds, fed_training_loss, label="Federated Edge", color='blue')
    plt.plot(rounds, edge_training_loss, label="Edge (Non-Federated)", color='orange')
    plt.plot(rounds, cloud_training_loss, label="Cloud-Based", color='green')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()

    # Testing Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(rounds, fed_testing_metrics['accuracy'], label="Federated Edge", color='blue')
    plt.plot(rounds, edge_testing_metrics['accuracy'], label="Edge (Non-Federated)", color='orange')
    plt.plot(rounds, cloud_testing_metrics['accuracy'], label="Cloud-Based", color='green')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Testing Accuracy")
    plt.title("Testing Accuracy Comparison")
    plt.legend()

    # Testing Latency
    plt.subplot(2, 2, 4)
    plt.plot(rounds, fed_testing_metrics['latency'], label="Federated Edge", color='blue')
    plt.plot(rounds, edge_testing_metrics['latency'], label="Edge (Non-Federated)", color='orange')
    plt.plot(rounds, cloud_testing_metrics['latency'], label="Cloud-Based", color='green')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Latency (s)")
    plt.title("Testing Latency Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()


import numpy as np
import tensorflow as tf
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

# =========================
# Step 1: Distribute Data Among Clients
# =========================

def distribute_data_among_clients(X, y_binary, y_continuous, num_clients):
    client_data = []
    for i in range(num_clients):
        idx = np.random.choice(len(X), len(X) // num_clients, replace=False)
        client_data.append((X[idx], y_binary[idx], y_continuous[idx]))
    return client_data

# =========================
# Step 2: Define the MLP Model for Binary Classification
# =========================

def create_mlp(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# =========================
# Step 3: Train MLP Federated Model
# =========================

def train_mlp_federated(client_data, input_shape, rounds=5):
    global_model = create_mlp(input_shape)
    start_time = time.time()

    for r in range(rounds):
        local_weights = []

        for X_local, y_binary_local, _ in client_data:
            local_model = create_mlp(input_shape)
            local_model.set_weights(global_model.get_weights())

            local_model.fit(X_local, y_binary_local, epochs=1, batch_size=32, verbose=0)
            local_weights.append(local_model.get_weights())

        # Aggregate weights by taking the mean across all clients
        new_weights = [np.mean(np.array(weights), axis=0) for weights in zip(*local_weights)]
        global_model.set_weights(new_weights)

        print(f"Round {r+1}/{rounds} completed.")

    end_time = time.time()
    latency = end_time - start_time

    return global_model, latency

# =========================
# Step 4: Train LASSO Federated Model
# =========================

def train_lasso_federated(client_data):
    global_coefficients = []

    for X_local, _, y_continuous_local in client_data:
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(X_local, y_continuous_local)
        global_coefficients.append(lasso_model.coef_)

    return np.mean(global_coefficients, axis=0)

# =========================
# Step 5: Evaluate Federated Models (Sensitivity, Specificity, Latency)
# =========================

'''
def evaluate_federated_models(mlp_model, lasso_coefficients, X_test, y_binary_test, y_continuous_test):
    # MLP Evaluation
    y_binary_pred = (mlp_model.predict(X_test) > 0.5).astype(int)
    binary_accuracy = accuracy_score(y_binary_test, y_binary_pred)

    # Calculate Sensitivity and Specificity
    cm = confusion_matrix(y_binary_test, y_binary_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate

    # LASSO Evaluation
    y_continuous_pred = X_test.dot(lasso_coefficients)
    continuous_mae = mean_absolute_error(y_continuous_test, y_continuous_pred)

    print("\nFederated Learning Model Evaluation:")
    print(f"Binary Classification Accuracy (MLP): {binary_accuracy}")
    print(f"Sensitivity (Recall) of MLP: {sensitivity}")
    print(f"Specificity of MLP: {specificity}")
    print(f"Continuous Regression MAE (LASSO): {continuous_mae}")
'''

import time
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, mean_squared_error
import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy loss function."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid division by zero errors
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def evaluate_federated_models(mlp_model, lasso_coefficients, X_test, y_binary_test, y_continuous_test):
    # --- MLP Evaluation (Binary Classification) ---
    start_time = time.time()
    y_binary_pred = (mlp_model.predict(X_test) > 0.5).astype(int)
    binary_latency = time.time() - start_time

    binary_accuracy = accuracy_score(y_binary_test, y_binary_pred)
    binary_loss = binary_cross_entropy_loss(y_binary_test, y_binary_pred)

    # Confusion Matrix for Sensitivity and Specificity
    cm = confusion_matrix(y_binary_test, y_binary_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    binary_metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'latency': binary_latency
    }

    # --- LASSO Evaluation (Continuous Regression) ---
    start_time = time.time()
    y_continuous_pred = X_test.dot(lasso_coefficients)
    continuous_latency = time.time() - start_time

    continuous_mae = mean_absolute_error(y_continuous_test, y_continuous_pred)
    continuous_loss = mean_squared_error(y_continuous_test, y_continuous_pred)

    continuous_metrics = {
        'latency': continuous_latency
    }

    # Print Metrics
    print("\nFederated Learning Model Evaluation:")
    print(f"Binary Model - Loss: {binary_loss}, Accuracy: {binary_accuracy}, Metrics: {binary_metrics}")
    print(f"Continuous Model - Loss: {continuous_loss}, MAE: {continuous_mae}, Metrics: {continuous_metrics}")


# =========================
# Step 6: Main Federated Learning Function
# =========================

def federated_learning_model(X, y_binary, y_continuous, num_clients=5, rounds=5):
    client_data = distribute_data_among_clients(X, y_binary, y_continuous, num_clients)

    # Train MLP Federated Model
    print("\nTraining MLP in Federated Setting...")
    mlp_model, mlp_latency = train_mlp_federated(client_data, input_shape=(X.shape[1],), rounds=rounds)

    # Train LASSO Federated Model
    print("\nTraining LASSO in Federated Setting...")
    lasso_coefficients = train_lasso_federated(client_data)

    return mlp_model, lasso_coefficients, mlp_latency

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, GRU, Flatten
from sklearn.metrics import accuracy_score, mean_squared_error
import random
import csv
from collections import deque

# Deep Q-Network (DQN) Model
def create_dqn_model(state_size, action_size):
    model = Sequential([
        Dense(64, input_dim=state_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_dqn_model(state_size, action_size)
        self.target_model = create_dqn_model(state_size, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Federated Learning with DQN
def federated_dqn_server(X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, communication_rounds, local_epochs, num_edge_devices):
    # Create models
    binary_model = create_binary_classification_model(input_shape=(X_train.shape[1], 1))
    continuous_model = create_continuous_regression_model(input_shape=(X_train.shape[1], 1))
    X_splits, y_binary_splits, y_continuous_splits = create_non_iid_data(X_train, y_binary_train, y_continuous_train, num_edge_devices)

    # Initialize DQN Agent
    state_size = num_edge_devices  # State: Resource metrics for all clients
    action_size = num_edge_devices  # Action: Which client to select
    dqn_agent = DQNAgent(state_size, action_size)

    # Metrics
    training_accuracy = []
    training_loss = []
    testing_metrics = {'binary': [], 'continuous': []}

    for t in range(communication_rounds):
        print(f"Communication Round {t+1}/{communication_rounds}")

        # Gather client resource metrics
        states = []
        for client_id in range(num_edge_devices):
            cpu, memory, latency = get_resource_metrics(client_id)
            states.append([cpu, memory, latency])
        states = np.array(states)

        # Select clients using DQN
        action = dqn_agent.act(states.reshape(1, -1))  # Select a client
        selected_client = action
        print(f"Selected Client: {selected_client}")

        # Local training
        binary_model_local = create_binary_classification_model(input_shape=(X_train.shape[1], 1))
        continuous_model_local = create_continuous_regression_model(input_shape=(X_train.shape[1], 1))
        binary_model_local.set_weights(binary_model.get_weights())
        continuous_model_local.set_weights(continuous_model.get_weights())

        binary_weights, binary_acc = local_training_binary(binary_model_local, X_splits[selected_client], y_binary_splits[selected_client], local_epochs)
        continuous_weights, continuous_mae = local_training_continuous(continuous_model_local, X_splits[selected_client], y_continuous_splits[selected_client], local_epochs)

        # Aggregate weights
        binary_model.set_weights(binary_weights)
        continuous_model.set_weights(continuous_weights)

        # Evaluate global models
        binary_loss, binary_acc, binary_metrics = evaluate_global_model(binary_model, X_test, y_binary_test, 'binary')
        continuous_loss, continuous_mae, continuous_metrics = evaluate_global_model(continuous_model, X_test, y_continuous_test, 'continuous')

        # Record metrics
        training_accuracy.append(binary_acc)
        training_loss.append(continuous_mae)
        testing_metrics['binary'].append(binary_metrics)
        testing_metrics['continuous'].append(continuous_metrics)

        print(f"Binary Model - Loss: {binary_loss}, Accuracy: {binary_acc}, Metrics: {binary_metrics}")
        print(f"Continuous Model - Loss: {continuous_loss}, MAE: {continuous_mae}, Metrics: {continuous_metrics}")


        # Store experience in DQN
        reward = binary_acc - binary_loss  # Example reward function
        next_state = states  # Simplified next state
        done = (t == communication_rounds - 1)
        dqn_agent.remember(states.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)

        # Replay experience
        if len(dqn_agent.memory) > 32:
            dqn_agent.replay(32)

        # Update target model
        dqn_agent.update_target_model()


    return binary_model, continuous_model



# Main Execution Flow
if __name__ == "__main__":

    file_path = '/content/drive/MyDrive/cleaned_data(1).csv'
    '''
    # Federated learning settings
    communication_rounds = 20
    learning_rate = 0.001
    local_epochs = 10
    num_edge_devices = 20
    privacy_param = 0.1

    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler = load_and_preprocess_data(file_path)

    # Add a dimension to fit LSTM input requirements
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Run Federated Learning
    binary_model, continuous_model = federated_server(
        X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test,
        communication_rounds, learning_rate, local_epochs, num_edge_devices, privacy_param
    )

    # Evaluate models
    binary_loss, binary_acc, binary_metrics = evaluate_global_model(binary_model, X_test, y_binary_test, 'binary')
    continuous_loss, continuous_mae, continuous_metrics = evaluate_global_model(continuous_model, X_test, y_continuous_test, 'continuous')

    print(f"Binary Model - Loss: {binary_loss}, Accuracy: {binary_acc}, Metrics: {binary_metrics}")
    print(f"Continuous Model - Loss: {continuous_loss}, MAE: {continuous_mae}, Metrics: {continuous_metrics}")

    # For Cloud Computing

    # Call this function after loading the data
    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler = load_and_preprocess_data(file_path)

    # Reshape the input data for RNN models
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cloud_metrics_binary, cloud_metrics_continuous = cloud_computing_model(
        X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test
    )

    # For Edge Computing without Federated Learning

    # Load and preprocess data
    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler = load_and_preprocess_data(file_path)

    # Simulate data across edge devices
    num_edges = 10
    X_train_split, y_binary_train_split, y_continuous_train_split = simulate_edge_data(X_train, y_binary_train, y_continuous_train, num_edges)
    X_test_split, y_binary_test_split, y_continuous_test_split = simulate_edge_data(X_test, y_binary_test, y_continuous_test, num_edges)

    # Reshape data for RNN models
    X_train_split = [x.reshape(x.shape[0], x.shape[1], 1) for x in X_train_split]
    X_test_split = [x.reshape(x.shape[0], x.shape[1], 1) for x in X_test_split]

    # Run edge computing model
    edge_metrics_binary, edge_metrics_continuous = edge_computing_model(
        X_train_split, X_test_split, y_binary_train_split, y_binary_test_split,
        y_continuous_train_split, y_continuous_test_split
    )
    '''

    # For MLP and Lasso based Federated Learning

    # Load and Preprocess Data
    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler = load_and_preprocess_data(file_path)

    # Run Federated Learning
    mlp_model, lasso_coefficients, latency = federated_learning_model(X_train, y_binary_train, y_continuous_train, num_clients=10, rounds=20)

    # Evaluate Federated Models
    evaluate_federated_models(
        mlp_model, lasso_coefficients, X_test, y_binary_test, y_continuous_test
    )
    print(f"MLP Training Latency: {latency:.4f} seconds")
    '''

    # For Federated Learning with DQN
    # Load and Preprocess Data
    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, scaler = load_and_preprocess_data(file_path)

    # Define communication_rounds and number of edge devices
    communication_rounds = 20
    num_edge_devices = 10
    local_epochs = 10

    # Call your training function
    federated_dqn_server(X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test, communication_rounds, local_epochs, num_edge_devices)


    '''



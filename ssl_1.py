import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import joblib


# Load the dataset
file_path = "C:/Users/kampa/Desktop/ML-Code/cleaned4_data.csv"
df = pd.read_csv(file_path, low_memory=False, skiprows=0)

# Convert DateAndTime to datetime format
df['DateAndTime'] = pd.to_datetime(df['DateAndTime'])

# Extract time-based features
df['Year'] = df['DateAndTime'].dt.year  # Extract year
df['Month'] = df['DateAndTime'].dt.month  # Extract month
df['Day'] = df['DateAndTime'].dt.day  # Extract day
df['Hour'] = df['DateAndTime'].dt.hour  # Extract 

# Drop the original timestamp column
df.drop(columns=['DateAndTime'], inplace=True)

# Define features and multi-target labels
target_columns = ['BE2-MJoistUp-MC', 'BE2-MJoistLo-MC','BN1-MJoistLo-MC','BN1-MJoistUp-MC','BN2-MJoistLo-MC','BN2-MJoistUp-MC',
                 'BN3-MJoistLo-MC','BN3-MJoistUp-MC','BS-MJoistRight-MC','BS-MJoistLeft-MC','BW1-MJoistUp-MC', 'BW1-MJoistLo-MC',]  # Multi-output targets
X = df.drop(columns=target_columns + ['BE1-MCheek', 'BE2-MCheek', 'BE1-TCheek', 'BE1-MCheek-MC','BE2-MCheek-MC',
                                      'BN1-MCheek-MC','BN2-MCheek-MC','BN3-MCheek-MC','BS-MCheek-MC','BE1-MJoistLo-MC',
                                      'BE1-MJoistUp-MC','BE1-MJoistLo-MC','FE-MJoistUp-MC', 'FE-MJoistLo-MC','FS-MJoistLeft-MC',
                                      'FS-MJoistRight-MC','FN-MJoistLeft-MC','FN-MJoistRight-MC','BW2-MJoistUp-MC',
                                      'BW2-MJoistLo-MC',
                                      'BN1-MCheek','BN2-MCheek','BN3-MCheek','BS-MCheek', 'BW1-MCheek', 'BW1-MCheek-MC',
                                      'BW2-MCheek', 'BW2-MCheek-MC','FN-MCheek', 'FN-MCheek-MC','FE-MCheek', 'FE-MCheek-MC',
                                      'FN-TPocket', 'FN-TCheek', 'FE-TPocket', 'FE-TCheek', 'FS-TPocket',
                                      'FN-RPocket-C', 'FN-RPocket-DP','FN-MJoistLeft', 'FN-MJoistRight', 'FE-MJoistUp',
                                      'FE-MJoistLo', 'FS-MJoistLeft', 'FS-MJoistRight',
                                      'FE-RPocket', 'FE-RPocket-C', 'FE-RPocket-DP', 'FS-RPocket',
                                      'BE2-TCheek' ,'BE2-RPocket','BE1-RPocket','Year', 'Month', 'Day', 'Hour',
                                      ]) # Features (excluding targets)
y = df[target_columns]  # Targets

# Split the data into labeled and unlabeled sets (20% labeled, 80% unlabeled)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_labeled = scaler.fit_transform(X_labeled)
X_unlabeled = scaler.transform(X_unlabeled)

# Initialize the multi-output XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Self-training loop
start_time = time.time()
for i in range(10):  # Number of iterations
    model.fit(X_labeled, y_labeled)

    # Predict on unlabeled data
    y_pred_unlabeled = model.predict(X_unlabeled)

    # Confidence measure: Mean absolute difference between predicted targets
    confidence = np.mean(np.abs(y_pred_unlabeled - np.mean(y_pred_unlabeled, axis=0)), axis=1)

    # Select the top 10% most confident samples
    top_indices = np.argsort(confidence)[:int(0.1 * len(X_unlabeled))]

    # Convert indices to actual DataFrame index values
    selected_indices = np.array(range(len(X_unlabeled)))[top_indices]

    # Add the most confident predictions to the labeled dataset
    X_labeled = np.vstack([X_labeled, X_unlabeled[selected_indices]])
    y_labeled = np.vstack([y_labeled, y_pred_unlabeled[selected_indices]])

    # Remove selected samples from the unlabeled dataset
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

training_time = time.time() - start_time

# Final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the final model on the fully labeled dataset
model.fit(X_train, y_train)

# Train the final model on the fully labeled dataset
start_time = time.time()
model.fit(X_train, y_train)
learning_time = time.time() - start_time

# Predict on the test set
y_pred = model.predict(X_test)

# Compute evaluation metrics per target
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Print results for each target variable
for i, col in enumerate(target_columns):
    print(f"Target: {col}")
    print(f"Mean Squared Error (MSE): {mse[i]:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse[i]:.4f}")
    #print(f"Mean Absolute Error (MAE): {mae[i]:.4f}")
    print(f"R² Score: {r2[i]:.4f}")
    print("-" * 30)

# Print training and learning time
print(f"Training Time: {training_time:.4f} seconds")
print(f"Learning Time: {learning_time:.4f} seconds")


# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)

# Define the Teacher Model
class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and train the teacher model
teacher_model = TeacherModel(X_train.shape[1], y_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=16):
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_mae += torch.mean(torch.abs(outputs - y_batch)).item()
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
                val_mae += torch.mean(torch.abs(val_outputs - y_val_batch)).item()
        
        history['loss'].append(running_loss / len(dataloader))
        history['mae'].append(running_mae / len(dataloader))
        history['val_loss'].append(val_loss / len(val_dataloader))
        history['val_mae'].append(val_mae / len(val_dataloader))
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history

# Train Teacher Model
history_teacher = train_model(teacher_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Generate pseudo-labels
teacher_model.eval()
with torch.no_grad():
    pseudo_labels = teacher_model(X_unlabeled_tensor)

# Combine labeled and pseudo-labeled data
X_combined = torch.cat((X_train_tensor, X_unlabeled_tensor), dim=0)
y_combined = torch.cat((y_train_tensor, pseudo_labels), dim=0)

# Define the Student Model
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and train the student model
student_model = StudentModel(X_combined.shape[1], y_combined.shape[1])
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Train Student Model
history_student = train_model(student_model, X_combined, y_combined, X_test_tensor, y_test_tensor)

# Save and Load Student Model
torch.save(student_model.state_dict(), "student_moisture_model.pth")

student_model = StudentModel(X_combined.shape[1], y_combined.shape[1])
student_model.load_state_dict(torch.load("student_moisture_model.pth"))
student_model.eval()

# Evaluate Student Model
y_pred = student_model(X_test_tensor).detach().numpy()
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')

# Print R² scores
for i, col in enumerate(target_columns):
    print(f"Target: {col}")
    print(f"R² Score: {r2_scores[i]:.4f}")
    print("-" * 30)

# Plot Training History
def plot_history(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(history['loss'], label='Train Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title(f'{title} - Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss (MSE)')
    ax[0].legend()
    
    ax[1].plot(history['mae'], label='Train MAE')
    ax[1].plot(history['val_mae'], label='Validation MAE')
    ax[1].set_title(f'{title} - Mean Absolute Error')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAE')
    ax[1].legend()
    
    plt.show()

# Plot for Teacher and Student Model
plot_history(history_teacher, "Teacher Model")
plot_history(history_student, "Student Model")

# Save the trained model
joblib.dump(model, 'student_model.pkl')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Contoh Data (Masukkan data Anda di sini atau ganti dengan hasil ekstraksi)
# Fitur: [Usia, Nilai, Jam Belajar]
# Target: 0 (Minat Rendah), 1 (Minat Tinggi)
data = np.array([
    [15, 75, 2],  # Usia, Nilai, Jam Belajar
    [16, 80, 3],
    [14, 60, 1],
    [17, 90, 4],
    [15, 85, 3],
    [14, 55, 1],
    [16, 70, 2],
    [17, 95, 4],
    [15, 65, 1],
    [16, 78, 3]
])
labels = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 1])  # Target minat belajar

# 2. Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 3. Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Bangun Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer 1
    Dense(8, activation='relu'),                                   # Hidden layer 2
    Dense(1, activation='sigmoid')                                # Output layer
])

# 5. Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Training model
model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.2)

# 7. Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}, Akurasi: {accuracy:.4f}")

# 8. Prediksi dengan data baru
new_data = np.array([[16, 85, 3]])  # Contoh data baru
new_data = scaler.transform(new_data)  # Normalisasi data baru
prediction = model.predict(new_data)
print(f"Prediksi minat belajar: {'Tinggi' if prediction[0] > 0.5 else 'Rendah'}")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed

# ---------------------------
# 1) Generate synthetic time series
def generate_series(n=1000):
    t = np.arange(n)
    trend = 0.01 * t
    seasonal = np.sin(2 * np.pi * t / 50)
    noise = 0.1 * np.random.randn(n)
    series = 10 + trend + seasonal + noise
    return series

series = generate_series(1000)

# ---------------------------
# 2) Scale data
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()

# ---------------------------
# 3) Windowing
INPUT_WINDOW = 30
OUTPUT_WINDOW = 10

def make_windows(data, input_w, output_w):
    X, Y = [], []
    for i in range(len(data) - input_w - output_w):
        X.append(data[i:i+input_w])
        Y.append(data[i+input_w:i+input_w+output_w])
    return np.array(X), np.array(Y)

X, Y = make_windows(series_scaled, INPUT_WINDOW, OUTPUT_WINDOW)
X = X[..., np.newaxis]
Y = Y[..., np.newaxis]

# Train/test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# ---------------------------
# 4) Build Seq2Seq LSTM model
# Encoder
encoder_inputs = Input(shape=(INPUT_WINDOW,1))
encoder_lstm = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(OUTPUT_WINDOW,1))
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(1))
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Prepare decoder input (teacher forcing)
decoder_input_train = np.zeros_like(Y_train)
decoder_input_test = np.zeros_like(Y_test)

# ---------------------------
# 5) Train
history = model.fit([X_train, decoder_input_train], Y_train,
                    validation_data=([X_test, decoder_input_test], Y_test),
                    epochs=20, batch_size=32, verbose=1)

# ---------------------------
# 6) Evaluate
preds = model.predict([X_test, decoder_input_test])
preds_unscaled = scaler.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
Y_test_unscaled = scaler.inverse_transform(Y_test.reshape(-1,1)).reshape(Y_test.shape)

mae = mean_absolute_error(Y_test_unscaled.flatten(), preds_unscaled.flatten())
rmse = np.sqrt(mean_squared_error(Y_test_unscaled.flatten(), preds_unscaled.flatten()))
print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")



# Output: Test MAE: 0.2083, Test RMSE: 0.2570

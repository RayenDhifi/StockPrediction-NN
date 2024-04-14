import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import rmsprop_v2
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler


stock = yf.Ticker("AAPL")

data = stock.history(period="1y") 

dates = []
open_prices = []
high_prices = []
close_prices = []
low_prices = []

# IMPORTING DATA

def get_data(data):
    for row in data.itertuples():
        date_str = str(row[0])
        open_prices_str = row[1]
        high_prices_str = row[2]
        close_prices_str = row[4]
        low_prices_str = row[3]
        dates.append(date_str)
        open_prices.append(float(open_prices_str))
        high_prices.append(float(high_prices_str))
        close_prices.append(float(close_prices_str))
        low_prices.append(float(low_prices_str))

    return

get_data(data)

# DATA ORGANISATION

d = np.array([dates])
op = np.array([open_prices])
cp = np.array([close_prices])
lp = np.array([low_prices])
hp = np.array([high_prices])

target_names = ['High', 'Low', 'Close']
data_target = np.vstack((d, op ,cp, lp, hp))
#DATA PREPERATION
shift_days = 1
shift_steps = shift_days * 1
data_target = data[target_names].shift(-shift_steps)
x_data = data.values[0:-shift_steps]
y_data = data_target.values[:-shift_steps]
num_data = len(x_data)
train_split = 0.9
num_train = int(train_split * num_data)
num_test = num_data - num_train

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

#SCALING DATA
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

#Data Generator
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

batch_size = 250
sequence_length = 223

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)

batch = 0
signal = 0
seq_in = x_batch[batch, :, signal]
seq_out = y_batch[batch, :, signal]


#Validation set
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

#Recurrent Neural Network (RNN using the Keras API)

model = Sequential()

model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))

if False:
    from tensorflow.python.keras.initializers import RandomUniform
    
    #ini-ranges must be lower sometimes
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

#Loss function
warmup_steps = 150
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    #calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss_fn = tf.losses.MeanSquaredError()
    loss = loss_fn(y_true=y_true_slice,
                   y_pred=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    
    return loss_mean

optimizer = rmsprop_v2.RMSProp(learning_rate=0.001)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)

#CallBack Functions
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
callback_tensorboard= TensorBoard(log_dir='.23_logs/',
                                  histogram_freq=0,
                                  write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factror=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


model.compile(loss='mse', optimizer='rmsprop')
'''
#To train the model, and generate your own checkpoint
model.fit(generator, epochs=100, steps_per_epoch=100,
          validation_data=validation_data, callbacks=callbacks)
'''

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))    

print("loss (test-set):", result)


def plot_combined_predictions(start_idx, length, future_steps=0, train=True):
    """
    Plot the combined predicted price curve along with the true data.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param future_steps: Number of future steps to predict.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train_scaled
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test_scaled
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # Combine predictions for open, high, low, and close into a single curve (e.g., average)
    combined_pred = np.mean(y_pred_rescaled[:, :4], axis=1)  # Taking only the first four signals
    
    # Reshape y_true to match the shape expected by the scaler
    y_true_reshaped = y_true.reshape(-1, num_y_signals)
    
    # Inverse scaling for true data
    y_true_rescaled = y_scaler.inverse_transform(y_true_reshaped)
    
    # Combine real data for open, high, low, and close into a single curve (e.g., average)
    combined_true = np.mean(y_true_rescaled[:, :4], axis=1) 
    
    # Extend time axis to accommodate future predictions
    extended_time_axis = np.arange(len(combined_true))
    
    # Make the plotting-canvas bigger.
    plt.figure(figsize=(15,5))
    
    # Plot the combined true data
    plt.plot(extended_time_axis, combined_true, label='True Price', color='orange')
    
    # Plot the combined predicted curve.
    plt.plot(np.arange(len(combined_true)), combined_pred, label='Predicted Price', color='blue')
    
    
    if future_steps > 0:
        # Predict future steps
        future_input = np.expand_dims(x[-1, -future_steps:, :], axis=0)
        future_pred = model.predict(future_input)
        future_pred_rescaled = y_scaler.inverse_transform(future_pred[0])
        combined_future_pred = np.mean(future_pred_rescaled[:, :4], axis=1)
        
        # Extend time axis to accommodate future predictions
        extended_future_time_axis = np.arange(len(combined_true), len(combined_true) + future_steps)
        
        # Plot the future predictions starting immediately after the last true data point
        plt.plot(np.arange(len(combined_true)-1, len(combined_true) + future_steps), [combined_true[-1]] + list(combined_future_pred), label='Future Prediction', color='green')
    
    # Plot labels etc.
    plt.ylabel('Price')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()



plot_combined_predictions(start_idx=400, length=500, future_steps=20, train=True)

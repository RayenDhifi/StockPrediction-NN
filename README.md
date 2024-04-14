
# Stock Price Prediction using Recurrent Neural Networks

This repository contains code for predicting stock prices using Recurrent Neural Networks (RNNs) implemented with TensorFlow/Keras. The model predicts future stock prices based on historical price data.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- yfinance
- NumPy
- Matplotlib
- Pandas
- scikit-learn

You can install these dependencies using pip:

```bash
pip install tensorflow yfinance numpy matplotlib pandas scikit-learn 
```
### Installation

To get started, clone this repository to your local machine using Git:
```bash
git clone https://github.com/RayenDhifi/StockPrediction-NN
cd StockPrediction-NN
```
### Running the Code
You can run the code by executing the StockPredictionNN.py script:
```bash
python StockPredictionNN.py
```
This script fetches historical stock price data for a specific ticker (in this case, "AAPL" for Apple Inc.), preprocesses the data, trains an RNN model, and makes predictions. You can modify the ticker symbol and other parameters directly in the script. 

```bash
stock = yf.Ticker("yourstock")
``` 

## Documentation
### Data Preparation
- Historical stock price data is fetched using the Yahoo Finance API (yfinance).
- The data is preprocessed, including feature scaling using MinMaxScaler.
### Model Architecture
- The RNN model architecture consists of a GRU layer followed by a dense layer.
- The loss function used is Mean Squared Error (MSE), with a warm-up period to ignore initial predictions.
- The optimizer used is RMSprop.
### Training and Evaluation
- The model is trained on a portion of the data and evaluated on a separate validation set.
- Callbacks such as early stopping and model checkpointing are used during training to prevent overfitting and save the best model.
In the repository, a checkpoint is provided to avoid having to train the model. If you're interested in training your model, you may delete the checkpoint and uncomment the model.fit function in line 209.
### Prediction and Visualization
After training, the model makes predictions on the test set and visualizes the predicted vs. true stock prices.
Future predictions can also be generated using the trained model.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/RayenDhifi/StockPrediction-NN/blob/main/LICENSE) file for details

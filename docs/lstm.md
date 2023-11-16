- [Basics of LSTM](#basics-of-lstm)
- [Basic Structure of LSTM](#basic-structure-of-lstm)
- [Step-by-Step Process](#step-by-step-process)
- [Training LSTMs](#training-lstms)
- [Imprelementing LSTMs](#imprelementing-lstms)
- [Application to Sales Anomaly Detection](#application-to-sales-anomaly-detection)
  - [LSTM for Regression Tasks](#lstm-for-regression-tasks)
    - [One Target Task](#one-target-task)
      - [1. Data Prepartion](#1-data-prepartion)
      - [2. Data Scaling](#2-data-scaling)
      - [3. LSTM Model Architecture](#3-lstm-model-architecture)
      - [4. Reshape Input Data](#4-reshape-input-data)
      - [5. Model Training](#5-model-training)
      - [6. Prediction](#6-prediction)
      - [7. Evalution](#7-evalution)
    - [Multiple Targets](#multiple-targets)
  - [Application Detection using LSTM Model for Regression](#application-detection-using-lstm-model-for-regression)
    - [1. Make Predictions](#1-make-predictions)
    - [2. Calculate Residuals](#2-calculate-residuals)


## Basics of LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is designed to capture long-term dependencies in data. It is particularly useful in tasks involving sequential data, such as time series prediction, natural language processing, and speech recognition.

## Basic Structure of LSTM

LSTMs have a more complex structure than traditional RNNs. They consist of memory cells and gates that control the flow of information into and out of the cells. The key components are:

1. Cell State (Ct): This is the memory of the LSTM. It can store information over long periods of time.

2. Hidden State (ht): This is the output of the LSTM. It is a filtered version of the cell state that only includes the relevant information.

3. Gates:

- Forget Gate (ft): Decides what information from the cell state should be thrown away or kept.
- Input Gate (it): Updates the cell state with new information.
- Cell Gate (C~t): Updates the cell state.
- Output Gate (ot): Produces the final output based on the cell state but filtered.

## Step-by-Step Process

1. Forget Gate Operation:
   - $f_t=\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
   - The forget gate determines what information from the cell state should be discarded.
2. Input Gate Operation
   - $i_t=\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
   - $\tilde{C}_t=\tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
   - The input gate updates the cell state with new information.
3. Update Cell State
   - $C_t=f_t\cdot C_{t-1} + i_t \cdot \tilde{C}_t$
   - The cell state is updated by combining the information from the forget and input gates.
4. Output Gate Operation
   - $o_t=\sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
   - $h_t=o_t \cdot \tanh(C_t)$
   - The output gate produces the final output based on the updated cell state.

## Training LSTMs

LSTMs are trained using backpropagation through time (BPTT), a variant of backpropagation. Gradient descent is used to minimize the error in the predicted output.

## Imprelementing LSTMs

Use Tensorflow or PyTorch.

## Application to Sales Anomaly Detection

Applying Long Short-Term Memory (LSTM) networks to sales anomaly detection, such as in the context of Chewy, can be a promising methodology. LSTMs are particularly well-suited for handling sequential data, making them useful in scenarios where the temporal aspect of the data is crucial, as is often the case with sales data.

LSTMs can capture long-term dependencies in sequential data. In the context of sales, where patterns might change over time due to seasons, promotions, or other external factors, the ability to model these dependencies is crucial for accurate anomaly detection.

Applying LSTM to sales anomaly detection for Chewy is a sound methodology, given its ability to handle temporal dependencies, irregular time intervals, and non-linear patterns. However, success depends on careful consideration of data quality, model complexity, and real-time processing requirements. Regular monitoring and fine-tuning are essential to ensure the model stays effective as sales patterns evolve over time.

### LSTM for Regression Tasks

#### One Target Task

Adapting LSTMs for regression with multiple features follows a similar structure to sequence-to-sequence tasks, but the key is to properly structure the input data and adjust the model architecture accordingly.

##### 1. Data Prepartion

Assuming you have a dataset with multiple features and a target variable (the variable you want to predict), organize your data into sequences. Each sequence should contain a set of features and the corresponding target value.

For example, we have `n` features, a sequence at time `t` might look $(X_{t-n}, X_{t-n+1}, ..., X_{t-1}, X_t, Y_t)$, where $X$ represents the features, and $Y$ represents the target variable.

##### 2. Data Scaling

Normalize or scale the features to bring them to a similar scale. This is important for the converhence and stability of the training process. We can use techniques like Min-Max scaling or Standardization (Z-score normalization).

##### 3. LSTM Model Architecture

Constructing the LSTM model involves specifying the input shape, the number of LSTM units, and the output layer for regresison. 

And the `units` parameter in the LSTM layer based on the complexity of your data and task. 

##### 4. Reshape Input Data

Reshape your input data to fit the LSTM model. The input shape should be `(num_samples, timesteps, n_featuers)`.

```python
import numpy as np

# Assuming X_train and y_train are your training data and labels
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
```

##### 5. Model Training

Train the LSTM model using the prepared data.

```python
model.fit(X_train_reshaped, y_train, epochs=num_epochs, batch_size=batch_size)
```

##### 6. Prediction

```python
# Assuming X_test is your test data
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

predictions = model.predict(X_test_reshaped)
```

##### 7. Evalution

Evaluate the performance of your model using regression metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE).

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
```

#### Multiple Targets

If you want to predict multiple target variables (Sales, Inventory, Ratings) simultaneously in a regression task and perform anomaly detection for each of them, you can create a multi-output LSTM model. Each output of the model will correspond to one of the target variables.


### Application Detection using LSTM Model for Regression

In anomaly detection using an LSTM model for regression, you can identify anomalies by comparing the model's predictions with the actual sales values. Anomalies occur when there are significant deviations between the predicted and actual values. Here's how you can find anomalies:

#### 1. Make Predictions

After training your LSTM model on the training data and validating it on a separate dataset, use the model to make predictions on your test set or any new data.

```python
# Assuming X_test is your test data
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, n_features))
predictions = model.predict(X_test_reshaped)
```

#### 2. Calculate Residuals

Calculate the residuals by subtracting the predicted values from the actual sales values. Residuals represent the errors or the differences between the predicted and actual values.

```python
residuals = y_test - predictions.flatten()
```

3. Define a Threshold

Set a threshold to distinguish between normal and anomalous behavior. This threshold can be based on statistical measures, such as standard deviations from the mean, or it can be determined through domain knowledge or experimentation.

```python
# For example, using standard deviation
threshold = np.std(residuals) * 3  # Adjust multiplier as needed
```

4. Identify Anomalies

```python
anomalies = np.abs(residuals) > threshold
```

5. Visualize Anomalies

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, residuals, label='Residuals')
plt.scatter(y_test.index[anomalies], residuals[anomalies], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()
```

Adjust the threshold and anomaly detection method based on the characteristics of your data and the desired sensitivity of your anomaly detection system. Fine-tuning may be necessary to strike the right balance between detecting anomalies and avoiding false positives.
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LSTM Model 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  
        return out

@st.cache_data
def load_data():
    merged_data_algorithm = pd.read_csv('merged_data_algorithm.csv')
    
    # Define top 10 products
    top_10_products = merged_data_algorithm['StockCode'].value_counts().index[:10].tolist()
    
    return merged_data_algorithm, top_10_products

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def main():
    st.title("Demand Forecasting with LSTM")

    merged_data_algorithm, top_10_products = load_data()

    st.sidebar.title("Select a Product")
    selected_product = st.sidebar.selectbox("Choose a product", top_10_products)

    st.sidebar.markdown("### Top 10 Products")
    for product in top_10_products:
        st.sidebar.markdown(f"- {product}")

  
    product_data = merged_data_algorithm[merged_data_algorithm['StockCode'] == selected_product]

    product_data['InvoiceDate'] = pd.to_datetime(product_data['InvoiceDate'])
    weekly_demand = product_data.resample('W-Mon', on='InvoiceDate').sum().reset_index()

    # Data normalization
    mean_quantity = weekly_demand['Quantity'].mean()
    std_quantity = weekly_demand['Quantity'].std()
    normalized_data = (weekly_demand['Quantity'] - mean_quantity) / std_quantity
    
    # sequences for LSTM
    sequence_length = 10  
    X, y = create_sequences(normalized_data.values, sequence_length)

    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    #tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, sequence_length, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, sequence_length, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    #parameters
    input_size = 1 
    hidden_size = 50  
    output_size = 1 

    model = LSTMModel(input_size, hidden_size, output_size)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   
    model.train()
    for epoch in range(300):  
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.view(-1), y_train_tensor)
        loss.backward()
        optimizer.step()

    #final loss
    st.write(f'Final Loss after training: {loss.item():.4f}')

    # Prediction
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()

    train_predictions_rescaled = train_predictions * std_quantity + mean_quantity
    test_predictions_rescaled = test_predictions * std_quantity + mean_quantity


    all_predictions_rescaled = np.full(weekly_demand['Quantity'].shape, np.nan)
    all_predictions_rescaled[sequence_length:train_size + sequence_length] = train_predictions_rescaled.flatten()
    all_predictions_rescaled[train_size + sequence_length:] = test_predictions_rescaled.flatten()


    dates = weekly_demand['InvoiceDate']

    
    train_errors = y_train - train_predictions.flatten()  # Calculate training errors
    test_errors = y_test - test_predictions.flatten()     # Calculate test errors


    plt.figure(figsize=(10, 5))
    plt.plot(dates, weekly_demand['Quantity'], label='Actual Demand', color='blue')
    plt.plot(dates, all_predictions_rescaled, label='Predicted Demand', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.title(f'Demand Forecasting for {selected_product}', fontsize=16)
    plt.legend()
    plt.grid()
    st.pyplot(plt)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training error histogram
    sns.histplot(train_errors, bins=20, ax=axes[0], kde=True, color='blue', alpha=0.7)
    axes[0].set_title('Training Error Distribution', fontsize=14)
    axes[0].set_xlabel('Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    # Test error histogram
    sns.histplot(test_errors, bins=20, ax=axes[1], kde=True, color='red', alpha=0.7)
    axes[1].set_title('Test Error Distribution', fontsize=14)
    axes[1].set_xlabel('Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    st.pyplot(fig)

if __name__ == "__main__":
    main()

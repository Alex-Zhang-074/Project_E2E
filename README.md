# README.md for GRU Factor Excavating Module

## Overview
This GRU (Gated Recurrent Unit) Factor Excavating Module started by replicating the results of the research report by Huatai Securities, Xiaoming Lin Team, 68th of the AI Series.Till now, it contains several basic neural network experiment notebooks and two versions of the GRU factor excavating module. The latest version takes the minute k-line data for a 5-day window into the model and expect to yield a daily factor aiming for the 5-day vwap-to-vwap return.

## File Structure
- Data loading and preprocessing
- GRU model definition using PyTorch
- Training and validation of the model
- Visualization of results
- Utility functions for data handling and processing


1. **Importing Libraries and Modules:**
   - NumPy
   - Pandas
   - Matplotlib
   - PyTorch
   - scikit-learn

2. **Data Handling and Preprocessing:**
   - We use minute k-line data here. First we eliminate the data of 14:58:00 and 14:59:00 as no transaction is made during this time. Then we concat the data within the previous five day window together and preprocess it before saving them as the model input. We will use all the feasible stocks for the date as a batch to train the model.

3. **GRU Model Setup:**
   - This model uses 2 layers of GRU with 50 hidden nodes. Then it will output the last prediction of each sequence and send them to a Dense layerand finally output the prediction result.

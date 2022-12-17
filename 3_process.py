import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow import keras

input_directory = 'flows/input'
model_file = 'models/'
scaler_file = 'scaler/scaler.save'
encoder_file = 'encoder/encoder.save'

scaler = joblib.load(scaler_file)
one_hot_encoder = joblib.load(encoder_file)
model = keras.models.load_model(model_file)

print(model.summary())
keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)

#removes all whitespaces from the CSVs in the startdir
for file in os.scandir(input_directory):
    if (file.path.endswith(".csv")):
        with open(file.path, 'r', encoding = "ISO-8859-1") as f:
            print("Reading from "+file.path)
            lines = f.readlines()
            lines = [line.replace(' ', '') for line in lines]
        with open(file.path, 'w') as f:
            f.writelines(lines)
            print("Wrote in "+file.path)

for file in os.scandir(input_directory):
    if (file.path.endswith(".csv")):
        print("Reading from "+file.path)
        df = pd.DataFrame()
        df = pd.read_csv(file.path, encoding = "ISO-8859-1")

        df.dropna(inplace=True)
        df.drop('FlowID', axis=1, inplace=True)
        print('Dropped NaN and FlowID')

        df.Timestamp = df.Timestamp.str[:8] + ' ' + df.Timestamp.str[8:]
        print('Fixed - added whitespace to timestamp')

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        print('Converted timestamp column to timestamp object')

        df = df.sort_values(by=['Timestamp'])
        df = df.reset_index(drop=True)
        print('Sorted by timestamp')

        # Convert SourcePort and DestinationPort in single Port Series
        df['Port'] = np.where((df['SrcPort'] <= df['DstPort']), df['SrcPort'], df['DstPort'])
        df['Port'] = np.where((df['Port'] == 8080) | (df['Port'] == 80), 80, df['Port'])
        df['Port'] = np.where((df['Port'] <= 4096), df['Port'], 4096)
        df.drop('SrcPort', axis=1, inplace=True)
        df.drop('DstPort', axis=1, inplace=True)
        print('Merged the Port columns')

        # Create column with Internal and External IP feature
        df['Source'] = np.where((df['SrcIP'].str.startswith('192.168.')) | (df['SrcIP'].str.startswith('172.16.'))| (df['SrcIP'].str.startswith('10.')), 'Internal', 'External')
        df['Destination'] = np.where((df['DstIP'].str.startswith('192.168.')) | (df['DstIP'].str.startswith('172.16.'))| (df['DstIP'].str.startswith('10.')), 'Internal', 'External')
        df.drop('SrcIP', axis=1, inplace=True) # perde a noção de ataques repetidos do mesmo sitio. corrigir com dicionario?
        df.drop('DstIP', axis=1, inplace=True)
        print('Converted SrcIP and Destination IP to Internal and External')

        df.drop('Timestamp', axis=1, inplace=True)

        df.drop('FlowByts/s', axis=1, inplace=True)
        df.drop('FlowPkts/s', axis=1, inplace=True)

        df.pop('Label')

        # Categorical encoding for non-output features
        list_columns = ['Protocol', 'Destination', 'Source', 'Port']
        print('Going to encode the columns', list_columns)
        transformed_data = one_hot_encoder.transform(df[list_columns])
        encoded_data = pd.DataFrame(transformed_data)
        df = pd.concat([df, encoded_data], axis=1)
        df.drop(columns=list_columns, inplace=True)

        df = scaler.fit_transform(df)

        x = np.array(df)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        predictions = model.predict(x)
        predictions = np.round(predictions).astype(int)
        print(predictions)
        np.savetxt('results/results_'+file.name, predictions, fmt='%d')

    print('Finished '+file.name+' from the folder '+input_directory)

print('All done!')
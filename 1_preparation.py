import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers import LSTM

startdir = 'flows/training-dataset'
flows_directory = 'flows/training-dataset'
dataframe_file = 'flows/dataframe.csv'
outcomes_file = 'flows/outcomes.csv'
model_file = 'models/'
scaler_file = 'scaler/scaler.save'
encoder_file = 'encoder/encoder.save'

#removes all whitespaces from the CSVs in the startdir
for file in os.scandir(startdir):
    if (file.path.endswith(".csv")):
        with open(file.path, 'r', encoding = "ISO-8859-1") as f:
            print("Reading from "+file.path)
            lines = f.readlines()
            lines = [line.replace(' ', '') for line in lines]
        with open(file.path, 'w') as f:
            f.writelines(lines)
            print("Wrote in "+file.path)

df = pd.DataFrame()

for file in os.scandir(flows_directory):
    if (file.path.endswith(".csv")):
        print("Reading from "+file.path)
        temp_df = pd.read_csv(file.path, encoding = "ISO-8859-1")
        df = pd.concat([df, temp_df])
print('Finished importing CSVs from the folder', flows_directory)

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
df['Port'] = np.where((df['SourcePort'] <= df['DestinationPort']), df['SourcePort'], df['DestinationPort'])
df['Port'] = np.where((df['Port'] == 8080) | (df['Port'] == 80), 80, df['Port'])
df['Port'] = np.where((df['Port'] <= 4096), df['Port'], 4096)
df.drop('SourcePort', axis=1, inplace=True)
df.drop('DestinationPort', axis=1, inplace=True)
print('Merged the Port columns')

# Create column with Internal and External IP feature
df['Source'] = np.where((df['SourceIP'].str.startswith('192.168.')) | (df['SourceIP'].str.startswith('172.16.'))| (df['SourceIP'].str.startswith('10.')), 'Internal', 'External')
df['Destination'] = np.where((df['DestinationIP'].str.startswith('192.168.')) | (df['DestinationIP'].str.startswith('172.16.'))| (df['DestinationIP'].str.startswith('10.')), 'Internal', 'External')
df.drop('SourceIP', axis=1, inplace=True) # perde a noção de ataques repetidos do mesmo sitio. corrigir com dicionario?
df.drop('DestinationIP', axis=1, inplace=True)
print('Converted SourceIP and Destination IP to Internal and External')

df.drop('Timestamp', axis=1, inplace=True)

df.drop('FlowBytes/s', axis=1, inplace=True)
df.drop('FlowPackets/s', axis=1, inplace=True)

df.drop('FwdHeaderLength', axis=1, inplace=True)

print('dataframe shape is', df.shape)

# Categorical encoding for non-output features
list_columns = ['Protocol', 'Destination', 'Source', 'Port']
print('Going to encode the columns', list_columns)
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

transformed_data = one_hot_encoder.fit(df[list_columns])
transformed_data = one_hot_encoder.transform(df[list_columns])
encoded_data = pd.DataFrame(transformed_data)
df = pd.concat([df, encoded_data], axis=1)
df.drop(columns=list_columns, inplace=True)
    
joblib.dump(one_hot_encoder, encoder_file)

# Categorical encoding for output
df['Label'] = np.where((df['Label'].str.startswith('BENIGN')), 'BENIGN', 'ATTACK')
ordinal_encoder = OrdinalEncoder()
outcomes = df.pop('Label')
outcomes = ordinal_encoder.fit_transform(outcomes.values.reshape(-1, 1))
print('Finished encoding the outcomes')

# Scaling input
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
joblib.dump(scaler, scaler_file)
print('Saved scaler params to disk')

np.savetxt(dataframe_file, df, delimiter=",")
np.savetxt(outcomes_file, outcomes, delimiter=",")

print('Saved dataframe to file in', dataframe_file)
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    # Load the data
    data_path = os.path.join('data', 'meta.csv')
    data = pd.read_csv(data_path)

    # Display the data
    print(data.head())
    print(data.info())
    print(data.describe())
    
    print(data['Grid'].value_counts())
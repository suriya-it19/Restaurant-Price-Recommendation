import pandas as pd
import os

input = "input"

data = pd.read_csv(os.path.join(input , "hotel2.csv"))

data.head()
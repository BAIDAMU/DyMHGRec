import dill
import pandas as pd

with open('./adj_diag_med.pkl', 'rb') as f:
    test_data = dill.load(f)

print(test_data)
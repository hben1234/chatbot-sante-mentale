import os
import pickle

# Specify the path to your file
file_path = r"C:\Users\houda\Desktop\website-med\backend\model\sentiment_modelNB.pkl"

# Print the absolute path of the file
absolute_path = os.path.abspath(file_path)
print(f"The absolute path of the file is: {absolute_path}")

# Check if the file exists
if os.path.exists(absolute_path):
    print("The file exists.")
else:
    print("The file does not exist.")
    exit()

# Check the first few bytes of the file to see if it looks like a pickle file
with open(absolute_path, 'rb') as file:
    file_start = file.read(10)
    print(f"The first 10 bytes of the file are: {file_start}")

# Attempt to load the file
try:
    with open(absolute_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except pickle.UnpicklingError:
    print("The file could not be unpickled. It may be corrupted or not a valid pickle file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

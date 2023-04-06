import os
import json

directory = 'data_split/Validation' # replace with the path to the directory you want to search

real_count = 0
fake_count = 0

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
    
    for filename, data in metadata.items():
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if data['label'] == 'REAL':
                real_count += 1
            elif data['label'] == 'FAKE':
                fake_count += 1

total_count = real_count + fake_count

real_percent = (real_count / total_count) * 100
fake_percent = (fake_count / total_count) * 100

print("Real Count: ", real_count)
print("Fake Count: ", fake_count)
print(f"Number of real videos: {real_count} ({real_percent:.2f}%)")
print(f"Number of fake videos: {fake_count} ({fake_percent:.2f}%)")

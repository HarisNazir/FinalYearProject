import json
import os

# Path to the directory containing the video files
video_dir = 'data_split/Train'

# Path to the JSON file
json_file = 'metadata.json'

# Read the JSON file and create a dictionary with file names and labels
with open(json_file, 'r') as f:
    data = json.load(f)

# Check if all file names in metadata.json correspond to files in the directory
metadata_files = set(data.keys())
directory_files = set(os.listdir(video_dir))
if metadata_files != directory_files:
    print(f'Error: Metadata file names do not match directory files')
    missing_files = metadata_files - directory_files
    if missing_files:
        print(f'The following files are in metadata.json but not in the directory:')
        print(missing_files)

labels = {}
filenames = set() # Keep track of unique file names
for filename, label_data in data.items():
    if 'label' in label_data:
        label = label_data['label']
        if label == 'REAL' or label == 'FAKE':
            # Check if file name already exists in the dictionary
            if filename in filenames:
                print(f'Error: Duplicate file name {filename}')
            else:
                labels[filename] = label
                filenames.add(filename)
        else:
            print(f'Error: Invalid label {label} for {filename}')
    else:
        print(f'Error: No label found for {filename}')
        # Remove the file if it does not have a label
        # os.remove(os.path.join(video_dir, filename))

# Count the number of real and fake videos
num_real = 0
num_fake = 0
for label in labels.values():
    if label == 'REAL':
        num_real += 1
    elif label == 'FAKE':
        num_fake += 1

# Print the results
print(f'Number of real videos: {num_real}')
print(f'Number of fake videos: {num_fake}')

# Calculate the percentage of real and fake videos
total = num_real + num_fake
if total > 0:
    pct_real = num_real / total * 100
    pct_fake = num_fake / total * 100
else:
    pct_real = 0
    pct_fake = 0

# Print the results
print(f'Percentage of real videos: {pct_real:.2f}%')
print(f'Percentage of fake videos: {pct_fake:.2f}%')

import os

data_fldf_path = os.path.join(os.getcwd(), 'data/imagenette2-320')
train_path = os.path.join(data_fldf_path, 'train')
test_path = os.path.join(data_fldf_path, 'val')

train_folders = os.listdir(train_path)
test_folders = os.listdir(test_path)

for folder in train_folders:
    files_path = os.path.join(train_path, folder)
    image_files = os.listdir(files_path)
    for f in image_files:
        fp = os.path.join(files_path, f)
        new_path = os.path.join(train_path, f)
        os.rename(fp, new_path)
        
for folder in test_folders:
    files_path = os.path.join(test_path, folder)
    image_files = os.listdir(files_path)
    for f in image_files:
        fp = os.path.join(files_path, f)
        new_path = os.path.join(test_path, f)
        os.rename(fp, new_path)


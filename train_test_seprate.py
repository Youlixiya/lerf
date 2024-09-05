# messy_rooms train_list & test list 
import os
import shutil

dataset_name = "messy_rooms"
scene_name = "large_corridor_25"

test_folder_path = f"data/{dataset_name}/{scene_name}/segmentations"
train_folder_path = f"data/{dataset_name}/{scene_name}/images"
train_list_file = f"data/{dataset_name}/{scene_name}/train_list.txt"
test_list_file = f"data/{dataset_name}/{scene_name}/test_list.txt"

to_delete = []

with open(test_list_file, "w") as f:
    for filename in os.listdir(test_folder_path):
        f.write(f'{filename}.png'+'\n')
        to_delete.append(filename.rstrip('\n'))

print('write test_list complete')

lines_to_keep = []

with open(train_list_file, "w") as f:
    for filename in os.listdir(train_folder_path):
        if filename.split('.')[0] not in to_delete:
            lines_to_keep.append(filename)

    lines_to_keep = [line +'\n' for line in lines_to_keep]    
    f.writelines(lines_to_keep)

print('write train_list complete')

source_file = test_list_file
destination_file = f"data/{dataset_name}/{scene_name}/val_list.txt"

shutil.copy(source_file, destination_file)
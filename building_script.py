import os
from sklearn.model_selection import KFold
import math
from code_to_country import *
import shutil
import random


data_path = "/Users/irischeng/Documents/GeoTrainr/data"
# creating a list of file paths to all files under data directory (this is the relative path btw)
files = os.listdir(data_path) 
random.shuffle(files)
total_files = len(files)
print('total files', total_files)

# now to split into train/test/validation, for now will do a simple 80/10/10 split
# that is, 80% of data will be used to train, 10% will be used to test, and remaining 10% validate
train_upper = int(math.floor(0.8 * total_files))
test_lower = train_upper
test_upper = int(math.floor(0.9 * total_files))
val_lower = test_upper

train_files = files[:train_upper]
test_files = files[test_lower:test_upper]
val_files = files[val_lower:]

all_files = [train_files, test_files, val_files]

# from csv file


# write script that creates a list of directories based off of country names....
# test, train, validate parent directories
# Directory


root_dir ="/Users/irischeng/Documents/GeoTrainr_data"
splits = ["train", "test", "val"]

curr_countries = get_list_of_countries()
country_dic = get_country_code_dic()



for dir in splits:
    dirpath = os.path.join(root_dir, dir)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

#split_dirs =[root_dir + f"/{split}" for split in splits]
split_dirs = ["/Users/irischeng/Documents/GeoTrainr_data/train",
              "/Users/irischeng/Documents/GeoTrainr_data/test",
              "/Users/irischeng/Documents/GeoTrainr_data/val"]

for country in curr_countries:
    for split in split_dirs:
        dirpath = os.path.join(split, country)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)


# for file_name in train_files:
#     file_name = file_name.decode('utf-8')
#     full_path = os.path.join(data_path, file_name)
#     country = file_name.split("_")[0]
#     full_country_name = country_dic[country]
#     new_path = f"/Users/irischeng/Documents/GeoTrainr_data/train/{full_country_name}/{file_name}"
#     if os.path.exists(new_path):
#         os.remove(new_path)
#     shutil.copyfile(full_path, new_path)

eu_dic = get_eu_dic()
def copy_files(info):
    country_count = {k:0 for k in list(eu_dic.keys())}
    for dir_file in info:
        dir, files = dir_file
        for file_name in files:
            full_path = os.path.join(data_path, file_name)
            country = file_name.split("_")[0]
            if country not in eu_dic:
                print('missing country file path', full_path)
                continue
            country_count[country] += 1
            full_country_name = country_dic[country]
            new_path = f"{dir}/{full_country_name}/{file_name}"
            if os.path.exists(new_path):
                os.remove(new_path)
            shutil.copyfile(full_path, new_path) 
    print(country_count)
    return country_count


copy_files(zip(split_dirs, all_files))
    






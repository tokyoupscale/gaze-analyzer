import string
from pathlib import Path
import os
import random
import shutil

# для генерации названий директорий 1 уровня список чаров
name_letters = list(string.ascii_uppercase)

lv1_dirs = []
lv2_dirs = []

"""
принцип построения уровней директории:
/{i}{y}
/{i}{y}{i+1}
/3{i+1}
"""

# параметр dataset_path - путь до og датасета
def shuffle_dataset(dataset_path):
    try:
        if not os.path.isdir("shuffled_dataset"):
            os.mkdir("../shuffled_dataset")   
            
        os.chdir("../shuffled_dataset")
        
        # ОЧЕНЬ ДОХУЯ ЗАНИМАЕТ МЕСТА СТОЛЬКО ПАПОК ПОКА ЧТО ТЕСТ РЕНДЖ 0, 3
        # for i in range(len(name_letters)):
        #     for y in range(len(name_letters)):

        # lv1, AA
        for i in range(0, 3):
            for y in range(0, 3):
                lv1_path = Path(f"{name_letters[i]}{name_letters[y]}")
                Path.mkdir(lv1_path)
                lv1_dirs.append(lv1_path)

                # lv2, aa{z}
                for z in range(0, 6):
                    lv2_path = Path.joinpath(lv1_path, f"{lv1_path.name.lower()}{z+1}")
                    Path.mkdir(lv2_path)
                    lv2_dirs.append(lv2_path)

                    # lv3, 3{x}
                    for x in range(0, 1):
                        lv3_path = Path.joinpath(lv2_path, f"3{x+1}")
                        Path.mkdir(lv3_path)

                        # files = os.listdir(dataset_path)
                        # files = [f for f in files if os.path.isfile(os.path.join(dataset_path, f))]
                        # files = random.sample(files, min(len(files), 1))

                        # for f in files:
                        #     source_file = os.path.join(dataset_path, f)
                        #     destination_file = os.path.join(lv3_path, f)
                        #     shutil.copy(source_file, destination_file)


    except FileExistsError as e:
        print("file already exists, try to delete them before using a function")

def unshuffle_dataset():
    pass
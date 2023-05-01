import random
import glob
import os
import shutil


def copyfiles(fil, root_dir):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]

    # copy image
    src_ = fil
    dest = os.path.join(root_dir, f"images\{filename}.PNG")
    shutil.copyfile(src_, dest)

    # copy annotations
    src = os.path.join(os.path.dirname(root_dir), f"labels\{filename}.txt")
    dest = os.path.join(root_dir, f"labels\{filename}.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dest)


root_path = r"..\datasets\random_small_objcts\union_recs_rescaled_1024"
image_dir = os.path.join(root_path, "images/")
label_dir = os.path.join(root_path, "labels/")
lower_limit = 0
files = glob.glob(os.path.join(image_dir, '*.PNG'))

random.shuffle(files)

folders = {"train": 0.8, "val": 0.1, "test": 0.1}
check_sum = sum([folders[x] for x in folders])

assert check_sum == 1.0, "Split proportion is not equal to 1.0"

for folder in folders:
    os.mkdir(os.path.join(root_path, folder))
    temp_label_dir = os.path.join(folder, "labels/")
    os.mkdir(os.path.join(root_path, temp_label_dir))
    temp_image_dir = os.path.join(folder, "images/")
    os.mkdir(os.path.join(root_path, temp_image_dir))

    limit = round(len(files) * folders[folder])
    for fil in files[lower_limit:lower_limit + limit]:
        copyfiles(fil, os.path.join(root_path, folder))
    lower_limit = lower_limit + limit
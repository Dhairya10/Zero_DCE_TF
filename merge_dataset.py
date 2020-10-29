import os
from tqdm import tqdm
import glob
import cv2

ROOT_FOLDER = '/home/alpha/Documents/Dataset_Part1'
dest_path = '/home/alpha/Documents/merged_dataset'

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

extensions=('*.png','*.jpg','*.jpeg','*.JPG','*.JPEG')
image_list = []
total_image_count = 0
sub_directories = os.listdir(ROOT_FOLDER)
for sub_directory in tqdm(sub_directories):
    print(sub_directory)
    image_list = []
    path = os.path.join(ROOT_FOLDER,sub_directory)
    os.chdir(path)
    for ext in extensions:
        image_list.extend(glob.glob(ext))
    for image in image_list:
        img = cv2.imread(image)
        try:
            cv2.imwrite(os.path.join(dest_path,f'{total_image_count}.jpg'),img)
            total_image_count +=1
        except Exception as e:
            print('Exception : ', e)
print('Image Count : ', total_image_count)



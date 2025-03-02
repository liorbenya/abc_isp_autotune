import os
import cv2
import brisque
from brisque.brisque import BRISQUE
import numpy as np
from PIL import Image
from threading import Lock
import csv

lock_brissque = Lock()
def calculate_brisque(image_path):
    # Read the image\
        
    img = Image.open(image_path)
    ndarray = np.asarray(img)
    lock_brissque.acquire()
    obj = BRISQUE(url=False)
    # Calculate the BRISQUE score for the image
    score = obj.score(img=ndarray)
    lock_brissque.release()
    # print(a[-1], "result is : ", score, flush=True)

    return score

def evaluate_images_in_folder(folder_path):
    # List all PNG files in the specified folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # Create a dictionary to store scores
    scores = {}
    
    # Loop through all images and calculate the BRISQUE score
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        score = calculate_brisque(image_path)
        scores[image_file] = score
        
        print(f"Image: {image_file}, BRISQUE Score: {score}")

    # Calculate the average BRISQUE score
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Average BRISQUE Score: {average_score}")
        return average_score
    else:
        print("No PNG images found in the folder.")
        return None
    
def main():
    # path = "C:/Users/liorby/Documents/src/cv-pipelines/Hailo_mercury_Rsimu_case/ResultsSweep/avg_images/corner_25/scene_corner_25_sharpen1_wdr0_dsmc_b200w200_dci_gamma2_ccm_scene_default.png"
    # path = "C:/Users/liorby/Downloads/ABCPython-master/results/res_config_0.0_350.0_1650.0_150.0_6.6_11.0.png"
    # path = "C:/Users/liorby/Downloads/ABCPython-master/results/res_config_0.0_50.0_1350.0_250.0_3.1_101.0.png"
    path = "C:/Users/hailo/src/abc_isp_autotune/results/res_90_config_511.0_511.0_100.0_50.0_2.6_91.0.png"
    # folder_path = "C:/Users/liorby/Documents/src/cv-pipelines/Hailo_mercury_Rsimu_case/ResultsSweep/avg_images/corner_25"
    # res = evaluate_images_in_folder(folder_path)
    res = calculate_brisque(path)
    print("result is : ", res, flush=True)
    
def evalute_report():
    csv_file_path = "C:/Users/hailo/src/abc_isp_autotune/report.csv"
    min_val = 0
    min_params = []
    with open(csv_file_path, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            if (float(line[-1]) > min_val):
                min_val = float(line[-1])
                min_params = line[:-1]
    print("the params: ", min_params)
    print("score: ", min_val)
if __name__ == '__main__':
    main()
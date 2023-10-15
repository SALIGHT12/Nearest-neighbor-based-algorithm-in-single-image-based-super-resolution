# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:21:50 2023

@author: SID AHMED Soumia
#Nearest neighbor-based algorithm in single-image based super-resolution:
"""
import os
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr  #soumia
from skimage.metrics import structural_similarity as compare_ssim
import imquality.brisque as brisque
# Define the scaling factor for resizing the image
scale_factor = 0.5
# Create a text file to store the SSIM and PSNR values
f = open('Neartst_PSNR_SSIM_RESULTS.txt', 'w')
# Get the absolute path of the current working directory
current_path = os.getcwd()
# Construct the path to the folder you want to access
folder_path = os.path.join(current_path, 'Data_Test')
result_path = os.path.join(current_path, 'RESULTS_NEAREST')
# Use the folder path to access the contents of the folder
files_in_folder = os.listdir(folder_path)

# create the noisy image directory if it doesn't exist
if not os.path.exists(result_path):
    os.makedirs(result_path)
for filename in os.listdir(folder_path):
    # check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # read the image
        Ireference = cv2.imread(os.path.join(folder_path, filename))
        score1= brisque.score(Ireference)
        # Get the size of the reference image
        nrows, ncols, np = Ireference.shape
        # Calculate the new size of the image after resizing
        new_size = (int(ncols * scale_factor), int(nrows * scale_factor))
        # Resize the image using bilinear interpolation
        Ilowres = cv2.resize(Ireference, new_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(result_path, "L_"+filename), Ilowres)
        # Resize the low-resolution image using nearest neighbor interpolation
        Inearest = cv2.resize(Ilowres, [nrows, ncols], interpolation=cv2.INTER_NEAREST)
        # save 
        cv2.imwrite(os.path.join(result_path, filename), Inearest)
        psnr_x_ = compare_psnr(Ireference, Inearest)
        ssim_x_ = compare_ssim(Ireference, Inearest, win_size=11, multichannel=True)
        score2= brisque.score(Inearest)
        base_name = os.path.splitext(filename)[0]
        print(f"{filename}: PSNR={psnr_x_:.2f}, SSIM={ssim_x_:.2f}, score1={score1:.2f},score2={score2:.2f}\n")
        f.write(f"{base_name}: PSNR={psnr_x_:.2f}, SSIM={ssim_x_:.2f}, score1={score1:.2f},score2={score2:.2f}\n")
################################AFFICHAGE###################################### 



#print("SSIM between Inference and Nearest Neighbor: {:.4f}".format(ssim_x_))
# # Display the figure
# plt.show()
# cv2.imwrite('C:/Users/Soumia/Desktop/Projet_Messali_Zoubeida/Soumia_programs/nearst/Inearest.jpg', Inearest)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 16))
Ireference = cv2.imread(os.path.join(folder_path,str(1) + ".jpg"))
Ilowres = cv2.imread(os.path.join(result_path, "L_" + str(1) + ".jpg"))
Inearest = cv2.imread(os.path.join(result_path,str(1) + ".jpg"))
axes[0].imshow(Ireference[:,:,::-1])
axes[1].imshow(Ilowres[:,:,::-1])
axes[2].imshow(Inearest[:,:,::-1])

# Set the titles of the subplots
axes[0].set_title("Original")
axes[1].set_title("Down")
axes[2].set_title("Recon")
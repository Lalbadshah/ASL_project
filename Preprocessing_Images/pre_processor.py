import cv2
import os
path = '/home/butterchicken/Documents/MLSP_Project_Code/mlsp_project/Preprocessing_Images/'
def load_images_from_folder(folder = path+'original_data/'):
    directory = path + 'Downscaled_Images/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        fx = img.shape[0] / 500
        fy = img.shape[1] / 500
        dimg = cv2.resize(img, None, 0, fx, fy)
        cv2.imwrite(directory + filename, dimg)



og_im = load_images_from_folder()


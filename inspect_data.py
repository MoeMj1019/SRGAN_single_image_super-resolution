import numpy as np
from matplotlib import pyplot as plt
import os

# inspect data for super resolution task (HR and LR images) / (results and test images)
def inspect_data(HR_path, LR_path, result_path=None, type_end='.png',
                 HR_imgs_prefix='' ,LR_imgs_prefix='', result_imgs_prefix='',
                 sample_images=3 ,imgs_indecies=None):
    '''
    inspect data for super resolution task (HR and LR images)
    args:
        HR_path : path to high resolution images
        LR_path : path to low resolution images
        type_end : type of images (default = '.png')
        sample_images : number of images to sample (default = 3)
    '''
    if sample_images <1:
        raise ValueError('sample_images must be greater than 0')
    
    directory_HR = os.fsencode(HR_path)
    directory_LW = os.fsencode(LR_path)

    files_HR = os.listdir(directory_HR)
    files_LR = os.listdir(directory_LW)
    files_HR = [os.fsdecode(file) for file in files_HR if os.fsdecode(file).endswith(type_end) 
                                                        and os.fsdecode(file).startswith(HR_imgs_prefix)]
    files_LR = [os.fsdecode(file) for file in files_LR if os.fsdecode(file).endswith(type_end) 
                                                        and os.fsdecode(file).startswith(LR_imgs_prefix)]
    files_LR.sort()
    files_HR.sort()
    
    shapes_HR = []
    shapes_LR = []

    for file_HR, file_LR in zip (files_HR, files_LR):
        img_HR = plt.imread(HR_path+file_HR)
        shapes_HR.append(img_HR.shape) 
        img_LR = plt.imread(LR_path+file_LR)
        shapes_LR.append(img_LR.shape)

    shapes_LR = np.array(shapes_LR)
    print('low resolution images : ')
    print('shapes : ', np.unique(shapes_LR, axis=0))
    print('all_of_same_shape : ' , np.all(shapes_LR == shapes_LR[0]))
    print('=========================')

    shapes_HR = np.array(shapes_HR)
    print('high resolution images : ')
    print('shapes : ', np.unique(shapes_HR, axis=0))
    print('all_of_same_shape : ' , np.all(shapes_HR == shapes_HR[0]))
    print('=========================')
 
    if result_path is not None:
        directory_result = os.fsencode(result_path)

        files_result = os.listdir(directory_result)
        files_result = [os.fsdecode(file) for file in files_result if os.fsdecode(file).endswith(type_end) 
                                                                    and os.fsdecode(file).startswith(result_imgs_prefix)]
        files_result.sort()
        
        shapes_result = []

        for file in files_result:
            img_result = plt.imread(result_path+file)
            shapes_result.append(img_result.shape) 

        shapes_result = np.array(shapes_result)
        print('result images : ')
        print('shapes : ', np.unique(shapes_result, axis=0))
        print('all_of_same_shape : ' , np.all(shapes_result == shapes_result[0]))
        print('=========================')


    if imgs_indecies is None:
        sample_images = np.min([sample_images, len(files_HR), len(files_LR)])
        imgs_indecies = np.random.choice(len(files_LR), sample_images, replace=False)
    else:
        sample_images = np.min([len(imgs_indecies), len(files_HR), len(files_LR)])

    # show sample original sized low resolution image
    img_LR = plt.imread(LR_path+files_LR[imgs_indecies[0]])
    plt.imshow(img_LR)
    plt.title('example of low resolution image')

    fig_scale = 5
    if result_path is not None:
        fig , axes = plt.subplots(sample_images,3,figsize=(fig_scale*3,fig_scale*sample_images))

        for idx, (ax1, ax2,ax3) in enumerate(axes):
            img_HR_name = files_HR[imgs_indecies[idx]]
            img_LR_name = files_LR[imgs_indecies[idx]]
            img_result_name = files_result[imgs_indecies[idx]]
            img_HR_title = os.path.basename(os.path.normpath(HR_path)) + '/' + img_HR_name
            img_LR_title = os.path.basename(os.path.normpath(LR_path)) + '/' + img_LR_name
            img_result_title = os.path.basename(os.path.normpath(result_path)) + '/' + img_result_name

            img_HR = plt.imread(HR_path+img_HR_name)
            img_LR = plt.imread(LR_path+img_LR_name)
            img_result = plt.imread(result_path+img_result_name)
            
            ax1.imshow(img_HR)
            ax2.imshow(img_result)
            ax3.imshow(img_LR)
            ax1.set_title(img_HR_title)
            ax2.set_title(img_result_name)
            ax3.set_title(img_LR_title)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(True)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            if idx == len(imgs_indecies)-1:
                ax1.xaxis.set_visible(True)
                ax2.xaxis.set_visible(True)
                ax3.xaxis.set_visible(True)
            if idx == 0:
                ax1.set_title('GT image')
                ax2.set_title('result image')
                ax3.set_title('LR image (cubic x4)')

    else:

        fig , axes = plt.subplots(sample_images,2,figsize=(fig_scale*2,fig_scale*sample_images))

        for idx, (ax1, ax2) in enumerate(axes):
            img_HR_name = files_HR[imgs_indecies[idx]]
            img_LR_name = files_LR[imgs_indecies[idx]]
            img_HR_title = os.path.basename(os.path.normpath(HR_path)) + '/' + img_HR_name
            img_LR_title = os.path.basename(os.path.normpath(LR_path)) + '/' + img_LR_name

            img_HR = plt.imread(HR_path+img_HR_name)
            img_LR = plt.imread(LR_path+img_LR_name)
            
            ax1.imshow(img_HR)
            ax2.imshow(img_LR)
            ax1.set_title(img_HR_title)
            ax2.set_title(img_LR_title)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(True)
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            if idx == len(imgs_indecies)-1:
                ax1.xaxis.set_visible(True)
                ax2.xaxis.set_visible(True)
            if idx == 0:
                ax1.set_title('GT image')
                ax2.set_title('LR image (cubic x4)')
    
    plt.tight_layout()
    plt.show()
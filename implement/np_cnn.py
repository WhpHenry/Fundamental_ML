# coding: utf-8

import sys
import skimage.data
import numpy as np
import matplotlib.pyplot as plt


def img_show(img, name=None):
    fig , ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img).set_cmap("gray")
    if name != None:
        plt.savefig(name)
    else:
        plt.show()
    plt.close(fig)

# each filter is a W
def set_filter(shape, window_count):
    def rand_filter(shape):
        f = np.array([np.random.randint(-1, 1, shape[1]) for _ in range(shape[0])])
        return f
    def vertical():
        return np.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
    def parallel():
        return np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]) 
    f = np.array([rand_filter(shape) for _ in range(window_count)])
    return f

# conv processing: return feature map of img
def conv(img, filters):
    if len(filters.shape) - len(img.shape) != 1:
        print("ERROR: filters need one more param than img shape")
        sys.exit() 
    if len(img.shape) > 2 or len(filters.shape) > 3:
        if img.shape[-1] != filters.shape[-1]:
            print("ERROR: Number of channels in both image and filter must match.")
            sys.exit()
    if filters.shape[1] != filters.shape[2]:
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')  
        sys.exit()
    if filters.shape[1] % 2 == 0:
            print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')  
            sys.exit()  
    
    def conv_(img, conv_filter):
        filter_size = conv_filter.shape[0]
        res = np.zeros(img.shape)
        for r in np.uint16(np.arange((filter_size/2), 
                                     (img.shape[0]-filter_size/2-2))):
            for c in np.uint16(np.arange((filter_size/2), 
                                         (img.shape[1]-filter_size/2-2))):
                curr_region = img[r:r+filter_size, c:c+filter_size]
                curr_res = curr_region * conv_filter
                conv_sum = np.sum(curr_res)
                res[r, c] = conv_sum
        res = res[filter_size//2:img.shape[0]-filter_size//2, 
                  filter_size//2:img.shape[1]-filter_size//2]
        return res

    # final result of features should be here
    fea_map = np.zeros((1+(img.shape[0]-filters.shape[1]),
                        1+(img.shape[1]-filters.shape[2]),
                        filters.shape[0]))
    for i in range(filters.shape[0]):
        if len(img.shape) > 2: # muti channel
            conv_map = conv_(img[:,:,0], filters[i,:,:,0])
            for j in range(1, img.shape[-1]):
                conv_map += conv_(img[:,:,j], filters[i,:,:,j])
        else:   # singal channel
            conv_map = conv_(img, filters[i])
        fea_map[:,:,i] = conv_map
    return fea_map 

# non-linear feature
def relu(fea):
    relu_opt = np.zeros(fea.shape)
    for fea_idx in range(fea.shape[-1]):
        for r in range(fea.shape[0]):
            for c in range(fea.shape[1]):
                relu_opt[r, c, fea_idx] = np.max(fea[r, c, fea_idx], 0)
    return relu_opt

def pooling(fea, size=2, stride=2):
    pool_opt = np.zeros((np.uint16((fea.shape[0]-size+1)/stride), 
                        np.uint16((fea.shape[1]-size+1)/stride), 
                        fea.shape[-1]))

    for fea_idx in range(fea.shape[-1]):
        pr = 0
        for r in np.arange(0, fea.shape[0] - size-1, stride):
            pc = 0
            for c in np.arange(0, fea.shape[1] - size-1, stride):
                pool_opt[pr, pc, fea_idx] = np.max(fea[r:r+size, c:c+size])
                pc += 1
            pr += 1
    return pool_opt

cat = skimage.data.chelsea()
coffee = skimage.data.coffee()
astronaut = skimage.data.astronaut()

img = skimage.color.rgb2gray(astronaut)
# img_show(img)
l1_filter = set_filter((3,3),2)
# vertical
l1_filter[0, :, :] = np.array([[[-1,0,1],[-1,0,1],[-1,0,1]]])
# parallel
l1_filter[1, :, :] = np.array([[[1,1,1],[0,0,0],[-1,-1,-1]]]) 
                                    
l1_conv = conv(img, l1_filter)
img_show(l1_conv[:,:,0]+l1_conv[:,:,1], 'l1_conv.png')
l1_relu = relu(l1_conv)
img_show((l1_relu[:,:,0]+l1_relu[:,:,1]), 'l1_relu.png')
l1_pool = pooling(l1_relu)
img_show((l1_pool[:,:,0]+l1_pool[:,:,1]), 'l1_pool.png')

l2_filter = np.random.rand(3, 5, 5, l1_pool.shape[-1])
l2_conv = conv(l1_pool, l2_filter)
img_show(l2_conv[:,:,0]+l2_conv[:,:,1], 'l2_conv.png')
l2_relu = relu(l2_conv)
img_show(l2_relu[:,:,0]+l2_relu[:,:,1], 'l2_relu.png')
l2_pool = pooling(l2_relu)
img_show((l2_pool[:,:,0]+l2_pool[:,:,1]), 'l2_pool.png')

l3_filter = np.random.rand(1, 7, 7, l2_pool.shape[-1])
l3_conv = conv(l2_pool, l3_filter)
l3_relu = relu(l3_conv)
l3_pool = pooling(l3_relu)
img_show(l3_pool[:,:,0], 'l3pool.png')

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
from matplotlib.animation import FFMpegWriter
from scipy import signal as sig

def harris(img, patch_size, kappa):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_y = sobel_x.T 
    Ix = sig.convolve2d(img, sobel_x, mode='same')
    Iy = sig.convolve2d(img, sobel_y, mode='same')
    print(Ix.shape, Iy.shape)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    IxIy = Ix * Iy
    print(Ixx.shape, Iyy.shape, IxIy.shape)
    # rectangular unifoem window
    patch_weights = np.ones((patch_size, patch_size))/patch_size**2
    sumIxx = sig.convolve2d(Ixx, patch_weights, mode='same')
    sumIyy = sig.convolve2d(Iyy, patch_weights, mode='same')
    sumIxIy = sig.convolve2d(IxIy, patch_weights, mode='same')
    detM = sumIxx*sumIyy - sumIxIy*sumIxIy
    traceM = sumIxx * sumIyy
    scores = detM - (kappa * traceM**2)
    scores[scores < 0] = 0
    print(scores.shape)
    return scores



corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
descriptor_radius = 9
nonmaximum_supression_radius = 8
match_lambda = 4

img = mpimg.imread('data/000000.png')
print(img.shape)

# Corner Response (Harris)
harris_score = harris(img, corner_patch_size, harris_kappa)

plt.imshow(harris_score, cmap=plt.get_cmap('gray'))
plt.show()
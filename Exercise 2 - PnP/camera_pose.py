import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter
import glob


def rgb2gray(rgb_image):
    """RGB image to Gray.
    Converts the RGB image to a Gray Scale image.
    Args:
        rgb_image: The RGB image to be converted.
    
    Returns:
        The Gray Scale  image of the input RGB image.
    """
    return np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])

def P3P(K, points_image, points_world):
    points_world_homogeneous = np.vstack([points_world, np.ones((1, points_world.shape[1]))])
    points_image_homogeneous = np.array([points_image[0][::2], 
                                         points_image[0][1::2],
                                         np.ones_like(points_image[0][::2])])
    xy_normalized = np.linalg.inv(K) @ points_image_homogeneous
    Q = np.zeros((points_world.shape[1]*2, 12))
    for i in range(points_world.shape[1]):
        Q[2*i, :4] = points_world_homogeneous[:, i]
        Q[2*i, -4:] = -xy_normalized[0, i] * points_world_homogeneous[:, i]
        Q[2*i+1, 4:8] = points_world_homogeneous[:, i]
        Q[2*i+1, -4:] = -xy_normalized[1, i] * points_world_homogeneous[:, i]
    U, D, V = np.linalg.svd(Q, full_matrices=False)
    M = -V[-1].reshape((3, 4))
    Ur, Dr, Vr = np.linalg.svd(M[:, :3], full_matrices=False)
    R = Ur @ np.eye(Vr.shape[0]) @ Vr
    alpha = np.linalg.norm(R) / np.linalg.norm(M[:, :3])
    tvec = M[:, -1].reshape((3, -1)) * alpha
    M_hat = np.hstack([R, tvec])
    return M_hat

def project_points(transformation, K, points_world):
    points_world_homogeneous = np.vstack([points_world, np.ones((1, points_world.shape[1]))])
    uv = K @ transformation @ points_world_homogeneous
    uv_normalized = np.array([[uv[0, :]/uv[2, :]], 
                              [uv[1, :]/uv[2, :]]]).squeeze()
    return uv_normalized 

fig = plt.figure(figsize=(13,5))

ax1 = fig.add_subplot(121, projection='3d')

K = np.loadtxt("data/K.txt")
points_world = np.loadtxt("data/p_W_corners.txt", delimiter=', ' or '').T
points_image = np.loadtxt("data/detected_corners.txt")
transformation_matrix = P3P(K, points_image, points_world)
uv = project_points(transformation_matrix, K, points_world)
pts_x = uv[0, :]
pts_y = uv[1, :]

ax1.scatter3D(xs = points_world[2, :], ys = points_world[0, :], zs = points_world[1, :])
# ax.invert_xaxis(), ax.invert_zaxis()
# ax.set_xlim(0, -50)
# ax.set_ylim(0, 50)
# ax.set_zlim(0, 50)
ax1.invert_xaxis(), ax1.invert_zaxis()
ax1.set_xlabel('Z axis')
ax1.set_ylabel('X axis')
ax1.set_zlabel('Y axis')
# ax.invert_yaxis()

img = mpimg.imread('data/images_undistorted/img_0001.jpg')
img = rgb2gray(img)

ax2 = fig.add_subplot(122)
ax2.imshow(img, cmap=plt.get_cmap('gray'))
ax2.scatter(points_image[0][0::2], points_image[0][1::2], marker='o', color='r')
ax2.scatter(pts_x, pts_y, marker='o', color='b')
plt.axis('off')
plt.show()
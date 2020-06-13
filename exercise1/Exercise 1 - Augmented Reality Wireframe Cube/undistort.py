import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def pose_vector_to_transformation_matrix(pose):
    omega = pose[0:3]       # shape : (1, 3)
    t = pose[3:6]           # shape : (1, 3)
    t = t.reshape(3, -1)    # shape : (3, 1)

    theta = np.linalg.norm(omega)
    k = omega / theta       # shape : (1, 3)
    k_hat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) # shape : (3, 3)
    R = np.eye(3) + np.sin(theta)*k_hat + (1 - np.cos(theta))*k_hat@k_hat    # shape : (3, 3)

    transformation_matrix = np.hstack([R, t])
    return transformation_matrix

def project_points(points3D, K, D):
    normalized_3D_points = np.array([[points3D[0, :]/points3D[2, :]], 
                                     [points3D[1, :]/points3D[2, :]],
                                     [points3D[2, :]/points3D[2, :]]]).squeeze()
    xyz_normalized = K @ normalized_3D_points
    uv = np.array([[xyz_normalized[0, :]/xyz_normalized[2, :]], 
                   [xyz_normalized[1, :]/xyz_normalized[2, :]]]).squeeze()
    return uv

def distort_points(normalized_points):
    xy = np.array([[normalized_points[0, :]/normalized_points[2, :]], 
                   [normalized_points[1, :]/normalized_points[2, :]]]).squeeze()
    r = np.sum(xy**2, axis=0)
    xy_d = (1 + (D[0]*r) + (D[1]*r**2)) * xy
    return xy_d.reshape(2, -1)

def project_points_distorted(points3D, K, D):
    normalized_3D_points = np.array([[points3D[0, :]/points3D[2, :]], 
                                     [points3D[1, :]/points3D[2, :]],
                                     [points3D[2, :]/points3D[2, :]]]).squeeze()
    xy_d = distort_points(normalized_3D_points)
    xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
    uv = K @ xy_d_normalized
    uv_normalized = np.array([[uv[0, :]/uv[2, :]], 
                              [uv[1, :]/uv[2, :]]]).squeeze()
    return uv

def undistort_image(img, K, D):
    undistorted_img = np.zeros_like(img)
    # print(undistorted_img.shape)
    for i in range(undistorted_img.shape[0]):
        for j in range(undistorted_img.shape[1]):
            xy_normalized = np.linalg.inv(K) @ np.vstack([j, i, 1.0])
            xy_d = distort_points(xy_normalized)
            xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
            # print(xy_d_normalized)
            uv = K @ xy_d_normalized
            uv_normalized = np.array([[uv[0, :]/uv[2, :]], 
                                      [uv[1, :]/uv[2, :]]]).reshape(2, -1)
            # print(uv_normalized.shape)
            u1 = int(np.floor(uv_normalized[0]))
            v1 = int(np.floor(uv_normalized[1]))
            if u1 > 0 and u1 <= img.shape[1] and v1 > 0 and v1 <= img.shape[0]:
                undistorted_img[i,j] = img[v1,u1]
    return undistorted_img

def undistort_image_vectorized(img, K, D):
    print(img.shape)
    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    u, v = np.meshgrid(x, y)
    u = u.reshape(1, -1)
    v = v.reshape(1, -1)
    uv_actual = np.vstack([u, v, np.ones_like(u)])
    xy_normalized = np.linalg.inv(K) @ uv_actual
    xy_d = distort_points(xy_normalized)
    xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
    uv_d = K @ xy_d_normalized
    uv_d_normalized = np.array([[uv_d[0, :]/uv_d[2, :]], 
                              [uv_d[1, :]/uv_d[2, :]]]).reshape(2, -1)
    u1 = np.floor(uv_d_normalized[0]).astype(int)
    v1 = np.floor(uv_d_normalized[1]).astype(int)
    undistorted_img = img[v1, u1].reshape(img.shape[0], img.shape[1])
    return undistorted_img


K = np.loadtxt('data/K.txt')
D = np.loadtxt('data/D.txt')
poses = np.loadtxt('data/poses.txt')

x = np.linspace(0, 32, 9)
y = np.linspace(0, 20, 6)
X, Y = np.meshgrid(x/100., y/100.)

img = mpimg.imread('data/images/img_0001.jpg')
img = rgb2gray(img)

transform_world2camera = pose_vector_to_transformation_matrix(poses[0])

X = X.reshape(1, -1)
Y = Y.reshape(1, -1)
Z = np.zeros_like(X, dtype=float)
ones_vector = np.ones_like(X, dtype=float)

points3D_world = np.vstack([X, Y, Z, ones_vector])
points3D_camera = transform_world2camera @ points3D_world
# points2D_image = project_points(points3D_camera, K, D)
# points2D_image = project_points_distorted(points3D_camera, K, D)

points2D_image = project_points(points3D_camera, K, D)
pts_x = points2D_image[0, :]
pts_y = points2D_image[1, :]
image_undistorted = undistort_image_vectorized(img, K, D)
plt.imshow(image_undistorted, cmap=plt.get_cmap('gray'))
plt.scatter(pts_x, pts_y, marker='o', color='r')
plt.show()
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
from matplotlib.animation import FFMpegWriter


metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)


def rgb2gray(rgb_image):
    """RGB image to Gray.

    Converts the RGB image to a Gray Scale image.

    Args:
        rgb_image: The RGB image to be converted.
    
    Returns:
        The Gray Scale  image of the input RGB image.
    """
    return np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])


def pose_vector_to_transformation_matrix(pose):
    """Pose Vector to Transformation Matrix.

    Converts the pose which has dimension of 1x6 into a 
    Transfomation Matrix of dimension of 3x4. The pose vector
    has the first 3 values as the rotations and the last 3 
    vectors as the translation of the camera. This Matrix 
    transforms the 3D world points(in homogeneous coordinates)
    into 3D points in camera coordinate system. 

    Args:
        pose: The vector composed of angle and translation
            vector of the camera.

    Returns:
        The transformation matrix which has a dimension of 
        3x4.
    """
    omega = pose[0:3]       # shape : (1, 3)
    t = pose[3:6]           # shape : (1, 3)
    t = t.reshape(3, -1)    # shape : (3, 1)
    theta = np.linalg.norm(omega)
    k = omega / theta       # shape : (1, 3)
    k_hat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) # shape : (3, 3)
    R = np.eye(3) + np.sin(theta)*k_hat + (1 - np.cos(theta))*k_hat@k_hat    # shape : (3, 3)
    transformation_matrix = np.hstack([R, t])   # shape : (3, 4)
    return transformation_matrix


def distort_points(points3D, D):
    """Distort Points.

    Applies radial distortion on the points so that the points
    can properly align with the distorted image. The radial distortion
    is given by 1 + k1*r^2 + k2*r^4 where r^2 = x^2 + y^2 is 
    called the radial component.

    Args:
        points3D: 3D points of dimension 3xn

    Return:
        Distorted 2D coordinates of dimension 2xn 
    """
    xy = np.array([[points3D[0, :]/points3D[2, :]], 
                   [points3D[1, :]/points3D[2, :]]]).squeeze()
    r = np.sum(xy**2, axis=0)
    xy_d = (1 + (D[0]*r) + (D[1]*r**2)) * xy
    return xy_d.reshape(2, -1)


def project_points(points3D, K):
    """Project points from 3D to image

    Project the points from the 3D camera coordinate frame to 
    the image plane using the camera intrinsics. Here we consider
    that there is no distortion in the image or the distortion has
    been compensated for.

    Args:
        points3D: The 3D points in the camera coordinate frame of
            dimension 3xn.
        K: The camera intrinsic matrix of dimension 3x3 which is of
            the form [[fx, 0.0, u],
                      [0.0, fy, v],
                      [0.0, 0.0, 1.0]].
    
    Returns:
        The final normalized coordinates in the image plane with 
        dimension of 2xn.
    """
    normalized_3D_points = np.array([[points3D[0, :]/points3D[2, :]], 
                                     [points3D[1, :]/points3D[2, :]],
                                     [points3D[2, :]/points3D[2, :]]]).squeeze()
    xyz_normalized = K @ normalized_3D_points
    uv = np.array([[xyz_normalized[0, :]/xyz_normalized[2, :]], 
                   [xyz_normalized[1, :]/xyz_normalized[2, :]]]).squeeze()
    return uv


def project_points_distorted(points3D, K, D):
    """Project Distorted Points

    This a more generalized procedure for projection of points from  
    the camera 3D coordinate frame to the image plane. The points are 
    first distorted in the way the image has been distorted so that 
    they align. Then these distorted points are projected to the image
    plane.

    Args:
        points3D: The 3D points in the camera coordinate frame of
            dimension 3xn.
        K: The camera intrinsic matrix of dimension 3x3 which is of
            the form [[fx, 0.0, u],
                      [0.0, fy, v],
                      [0.0, 0.0, 1.0]].
        D: the camera distortion parameters which are ususlly of the 
            form [k1, k2, k3, p1, p2].

    Returns:
        The final normalized coordinates in the image plane with 
        dimension of 2xn.
    """
    normalized_3D_points = np.array([[points3D[0, :]/points3D[2, :]], 
                                     [points3D[1, :]/points3D[2, :]],
                                     [points3D[2, :]/points3D[2, :]]]).squeeze()
    xy_d = distort_points(normalized_3D_points, D)
    xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
    uv = K @ xy_d_normalized
    uv_normalized = np.array([[uv[0, :]/uv[2, :]], 
                              [uv[1, :]/uv[2, :]]]).squeeze()
    return uv


def undistort_image_vectorized(img, K, D):
    """Image Undistortion

    Performs undistortion on the distorted image using the distortion
    coefficients. The procedure is to consider a grid (in image plane) 
    of dimension which is equal to the dimension of the image. Using the 
    K matrix we can recover the normalized xy coordinates in the camera
    frame. These points are then distorted using D and projected using 
    K into the image plane. Finally, a mapping from distorted image plane
    to the rectified image can be obtained.

    Args:
        img: The image to be undistorted.
        K: The camera intrinsic matrix of dimension 3x3 which is of
            the form [[fx, 0.0, u],
                      [0.0, fy, v],
                      [0.0, 0.0, 1.0]].
        D: the camera distortion parameters which are ususlly of the 
            form [k1, k2, k3, p1, p2].

    Returns:
        The undistorted image.
    """
    x = np.linspace(0, img.shape[1]-1, img.shape[1])
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    u, v = np.meshgrid(x, y)
    u = u.reshape(1, -1)
    v = v.reshape(1, -1)
    uv_actual = np.vstack([u, v, np.ones_like(u)])
    xy_normalized = np.linalg.inv(K) @ uv_actual
    xy_d = distort_points(xy_normalized, D)
    xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
    uv_d = K @ xy_d_normalized
    uv_d_normalized = np.array([[uv_d[0, :]/uv_d[2, :]], 
                              [uv_d[1, :]/uv_d[2, :]]]).reshape(2, -1)
    u1 = np.floor(uv_d_normalized[0]).astype(int)
    v1 = np.floor(uv_d_normalized[1]).astype(int)
    undistorted_img = img[v1, u1].reshape(img.shape[0], img.shape[1])
    return undistorted_img


def undistort_image(img, K, D):
    """
    This is another method to undistort the image but is very slow
    as it consider each pont separately and not in a vectorized 
    manner. The procedure is similar to the above.
    """
    undistorted_img = np.zeros_like(img)
    for i in range(undistorted_img.shape[0]):
        for j in range(undistorted_img.shape[1]):
            xy_normalized = np.linalg.inv(K) @ np.vstack([j, i, 1.0])
            xy_d = distort_points(xy_normalized, D)
            xy_d_normalized = np.vstack([xy_d, np.ones((1, xy_d.shape[1]))])
            uv = K @ xy_d_normalized
            uv_normalized = np.array([[uv[0, :]/uv[2, :]], 
                                      [uv[1, :]/uv[2, :]]]).reshape(2, -1)
            u1 = int(np.floor(uv_normalized[0]))
            v1 = int(np.floor(uv_normalized[1]))
            if u1 > 0 and u1 <= img.shape[1] and v1 > 0 and v1 <= img.shape[0]:
                undistorted_img[i,j] = img[v1,u1]
    return undistorted_img


# Loading teh camera intrinsics and the parameters
K = np.loadtxt('data/K.txt')
D = np.loadtxt('data/D.txt')
poses = np.loadtxt('data/poses.txt')

"""
Projecting a Cube onto the grid using the camera poses and the 
camera intrinsics.
"""
all_image_paths = sorted(glob.glob('data/images/*'))
fig = plt.figure()
with writer.saving(fig, "1.mp4", 100):
    for i in range(len(all_image_paths)):
        img = rgb2gray(mpimg.imread(all_image_paths[i]))
        transform_world2camera = pose_vector_to_transformation_matrix(poses[i])
        cube_points = np.array([[0.00, 0.00, 0.00, 1.0],
                                [0.00, 0.08, 0.00, 1.0],
                                [0.08, 0.08, 0.00, 1.0],
                                [0.08, 0.00, 0.00, 1.0],
                                [0.00, 0.00, -0.08, 1.0],
                                [0.00, 0.08, -0.08, 1.0],
                                [0.08, 0.08, -0.08, 1.0],
                                [0.08, 0.00, -0.08, 1.0]]).T

        points3D_camera = transform_world2camera @ cube_points
        points2D_image = project_points(points3D_camera, K)
        pts_x = points2D_image[0, :]
        pts_y = points2D_image[1, :]
        image_undistorted = undistort_image_vectorized(img, K, D)
        plt.axis([0, img.shape[1], img.shape[0], 0])
        plt.axis('off')
        plt.imshow(image_undistorted, cmap=plt.get_cmap('gray'))
        plt.scatter(pts_x, pts_y, marker='o', color='r')
        plt.plot(pts_x[[0,1,2,3,0]], pts_y[[0,1,2,3,0]], 'r')
        plt.plot(pts_x[[4,5,6,7,4]], pts_y[[4,5,6,7,4]], 'r')
        plt.plot(pts_x[[1,5]], pts_y[[1,5]], 'r')
        plt.plot(pts_x[[0,4]], pts_y[[0,4]], 'r')
        plt.plot(pts_x[[2,6]], pts_y[[2,6]], 'r')
        plt.plot(pts_x[[3,7]], pts_y[[3,7]], 'r')
        writer.grab_frame()
        plt.clf()
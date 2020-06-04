import numpy as np 
import cv2
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import FFMpegWriter


# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# writer = FFMpegWriter(fps=30, metadata=metadata)


def cross_product_matrix(k):
    k_hat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return k_hat


def project_points(R, t, X, Y, uv):
    transformation = np.hstack([R, t])
    pixels = K @ transformation @ np.array([[X, Y, 0, 1]]).T
    uv.append(np.array([pixels[0] / pixels[2], pixels[1] / pixels[2]]).T)
    return uv


def project_points_distorted(R, D, t, X, Y, uv):
    transformation = np.hstack([R, t])
    XYZ_C = transformation @ np.array([[X, Y, 0, 1]]).T
    xy = np.array([XYZ_C[0] / XYZ_C[2], XYZ_C[1] / XYZ_C[2]])
    r = (xy[0]*xy[0]) + (xy[1]*xy[1])
    xy_d = (1 + (D[0]*r) + (D[1]*r**2)) * xy
    pixels = K @ np.vstack([xy_d, 1.0])
    uv.append(np.array([pixels[0] / pixels[2], pixels[1] / pixels[2]]).T)
    return uv


K = np.loadtxt('data/K.txt')
D = np.loadtxt('data/D.txt')
poses = np.loadtxt('data/poses.txt')

x = np.linspace(0, 32, 9)
y = np.linspace(0, 20, 6)
X, Y = np.meshgrid(x/100., y/100.)

img = cv2.imread('data/images/img_0001.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

omega = poses[0, 0:3]
theta = np.linalg.norm(omega)
k = omega / theta
k_hat = cross_product_matrix(k)
R = np.eye(3) + np.sin(theta)*k_hat + (1 - np.cos(theta))*k_hat@k_hat
t = poses[0, 3:6]
t = t.reshape(3, -1)

uv = []
for i in range(6):
    for j in range(9):
        uv = project_points_distorted(R, D, t, X[i, j], Y[i, j], uv)
        # uv = project_points(R, t, X[i, j], Y[i, j], uv)
uv = np.squeeze(np.array(uv))
pts_x = uv[:, 0]
pts_y = uv[:, 1]
print(pts_x.shape)

implot = plt.imshow(img)
plt.scatter(pts_x, pts_y, marker='o', color='r')
plt.show()



# all_images = sorted(glob.glob("data/images/*"))
# # print(len(all_images))
# # print(all_images[0:20])
# fig = plt.figure()
# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(50):
#         print(i)
#         img = cv2.imread(all_images[i])
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         omega = poses[i, 0:3]
#         theta = np.linalg.norm(omega)
#         k = omega / theta
#         k_hat = cross_product_matrix(k)
#         R = np.eye(3) + np.sin(theta)*k_hat + (1 - np.cos(theta))*k_hat@k_hat
#         t = poses[i, 3:6]
#         t = t.reshape(3, -1)

#         # uv = []
#         # for i in range(6):
#         #     for j in range(9):
#         #         transformation = np.hstack([R, t])
#         #         pixels = K @ transformation @ np.array([[X[i, j], Y[i, j], 0, 1]]).T
#         #         uv.append(np.array([pixels[0] / pixels[2], pixels[1] / pixels[2]]).T)
#         # uv = np.squeeze(np.array(uv))
#         # pts_x = uv[:, 0]
#         # pts_y = uv[:, 1]

#         # implot = plt.imshow(img)
#         # plt.scatter(pts_x, pts_y, marker='o', color='r')
#         # plt.show()

#         cube_pts = np.array([[0.00, 0.00, 0.00, 1.0],
#                             [0.00, 0.08, 0.00, 1.0],
#                             [0.08, 0.08, 0.00, 1.0],
#                             [0.08, 0.00, 0.00, 1.0],
#                             [0.00, 0.00, -0.08, 1.0],
#                             [0.00, 0.08, -0.08, 1.0],
#                             [0.08, 0.08, -0.08, 1.0],
#                             [0.08, 0.00, -0.08, 1.0]])
#         uv = []
#         transformation = np.hstack([R, t])
#         cube_pixels = K @ transformation @ cube_pts.T 
#         cube_pixels = np.array([cube_pixels[0, :] / cube_pixels[2, :], cube_pixels[1, :] / cube_pixels[2, :]])

#         implot = plt.imshow(img)
#         plt.scatter(cube_pixels[0, :], cube_pixels[1, :], marker='o', color='r')
#         plt.plot(cube_pixels[0, [0,1,2,3,0]], cube_pixels[1, [0,1,2,3,0]], 'r')
#         plt.plot(cube_pixels[0, [4,5,6,7,4]], cube_pixels[1, [4,5,6,7,4]], 'r')
#         plt.plot(cube_pixels[0, [1,5]], cube_pixels[1, [1,5]], 'r')
#         plt.plot(cube_pixels[0, [0,4]], cube_pixels[1, [0,4]], 'r')
#         plt.plot(cube_pixels[0, [2,6]], cube_pixels[1, [2,6]], 'r')
#         plt.plot(cube_pixels[0, [3,7]], cube_pixels[1, [3,7]], 'r')
#         writer.grab_frame()
#         plt.clf()
#         # plt.show()
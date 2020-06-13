# Augmented reality wireframe cube

The goal of this exercise is to superimpose a virtual cube on a video of a planar grid viewed from different orientations. This also helps to familiarize with the basics of perspective projection, change of coordinate systems and lens distortion, as well as basic image processing.

## Projecting Points on the Image

Consider the distorted image:

<img src="./data/images/img_0001.jpg" alt="drawing" width="376" height="240"/>

Projecting the points on the above image,

|       Projection of Points before Distortion of Points       |       Projection of Points after Distortion of Points        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./Outputs/projection_without_distort_points.png" alt="drawing" width="376" height="240"/> | <img src="./Outputs/projection_after_distort_points.png" alt="drawing" width="376" height="240"/> |



## Undistorting the Image

|                       Distorted Image                        |                      Undistorted Image                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./data/images/img_0001.jpg" alt="drawing" width="376" height="240"/> | <img src="./Outputs/undistort_1.png" alt="drawing" width="376" height="240"/> |

Projecting the points after Undistorting the image,

<img src="./Outputs/projection_after_undistorting_image.png" alt="drawing" width="376" height="240"/>



## Projecting a cube on the Image

[![Project Virtual Cube on Grid](http://img.youtube.com/vi/qwMa-T378S8/0.jpg)](http://www.youtube.com/watch?v=qwMa-T378S8 "Project Virtual Cube on Grid")


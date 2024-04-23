import torch
import numpy as np
import torch.nn.functional as F
def convert_depth_image_to_point_cloud(depth_img,intrinsic_matrix,extrinsic_matrix):
    # Convert the depth image to a point cloud
    height, width = depth_img.shape
    point_cloud = np.zeros((3, height, width))

    for v in range(height):
        for u in range(width):
            z = depth_img[v, u]
            x = (u - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
            y = (v - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
            homogeneous_point = np.dot(extrinsic_matrix, np.array([x, y, z, 1]))
            point_cloud[:, v, u] = homogeneous_point[:3] / homogeneous_point[3]
    return point_cloud

def resize_point_cloud(depth_image):
    point_cloud_tensor = torch.from_numpy(depth_image)

    # Add an extra dimension for the batch size
    point_cloud_tensor = point_cloud_tensor.unsqueeze(0)
    # Resize the tensor
    resized_point_cloud_tensor = F.interpolate(point_cloud_tensor, size=(224, 224), mode='bilinear')
    return resized_point_cloud_tensor.to(torch.float32)
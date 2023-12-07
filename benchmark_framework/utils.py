import cv2
import numpy as np


def best_fit(src, dst):
    dst_mean = np.mean(dst, axis=0)
    src_mean = np.mean(src, axis=0)

    src_vec = (src - src_mean).flatten()
    dst_vec = (dst - dst_mean).flatten()

    a = np.dot(src_vec, dst_vec) / np.linalg.norm(src_vec) ** 2
    b = 0
    for i in range(dst.shape[0]):
        b += src_vec[2 * i] * dst_vec[2 * i + 1] - src_vec[2 * i + 1] * dst_vec[2 * i]
    b = b / np.linalg.norm(src_vec) ** 2

    tf_matrix = np.array([[a, b], [-b, a]])
    src_mean = np.dot(src_mean, tf_matrix)

    return tf_matrix, dst_mean - src_mean


def align_face(image, points_src, points_dst, dst_image_size):

    a, t = best_fit(points_src, points_dst)
    tf = np.concatenate((a, t.reshape(1, 2)), axis=0)

    aligned_image = cv2.warpAffine(
        image,
        tf.T,
        (dst_image_size, dst_image_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS,
        borderMode=cv2.BORDER_CONSTANT
    )

    return aligned_image


def get_aligned_face_crop(image, points, output_size):

    detected_points = np.array([
        [(points[44][0] + points[54][0]) / 2, (points[44][1] + points[54][1]) / 2],
        [(points[43][0] + points[53][0]) / 2, (points[43][1] + points[53][1]) / 2],
        [points[9][0], points[9][1]],
        [points[66][0], points[66][1]],
        [points[65][0], points[65][1]]
    ], dtype=np.float32)

    reference_points = np.array(
        [[0.33623695, 0.31037876],
         [0.65085478, 0.30863858],
         [0.49454586, 0.48931003],
         [0.36529677, 0.67349663],
         [0.6258378, 0.67205556]],
        dtype=np.float32) * output_size

    aligned_crop = align_face(
        image=image,
        points_src=detected_points,
        points_dst=reference_points,
        dst_image_size=output_size
    )

    return aligned_crop

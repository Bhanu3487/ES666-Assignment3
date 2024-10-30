import glob
import cv2
import os
import numpy as np

class PanaromaStitcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Total {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            print("Not enough images to stitch.")
            return None, []

        stitched_image = None
        homography_matrix_list = []

        for i in range(len(all_images)):
            print(f"Processing image {i + 1}/{len(all_images)}: {all_images[i]}")
            curr_img = cv2.imread(all_images[i])
            if curr_img is None:
                print(f"Error reading image: {all_images[i]}")
                continue

            curr_img = cv2.resize(curr_img, (640, 480))  # Resize for memory efficiency

            if stitched_image is None:
                stitched_image = curr_img
                print(f"First image set as the initial stitched image.")
                continue

            # Detect and compute features
            prev_kp, prev_desc = self.detect_and_compute(stitched_image)
            curr_kp, curr_desc = self.detect_and_compute(curr_img)

            print(f"Found {len(prev_kp)} keypoints in the previous image and {len(curr_kp)} keypoints in the current image.")

            # Match features
            matches = self.match_features(prev_desc, curr_desc)
            print(f"Found {len(matches)} matches.")

            # Find homography matrix if enough matches are found
            if len(matches) > 4:  # At least 4 matches required
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

                if homography_matrix is not None:
                    homography_matrix_list.append(homography_matrix)
                    print(f"Homography matrix computed for images {i-1} and {i}.")

                    # Warp and stitch the current image
                    stitched_image = self.warp_and_stitch(stitched_image, curr_img, homography_matrix)
                    print(f"Images {i-1} and {i} stitched successfully.")
                else:
                    print(f"Homography computation failed for images {i-1} and {i}.")
            else:
                print(f"Not enough matches found between image {i-1} and {i}. Skipping image {i}.")

            # Clear the current image from memory
            del curr_img

        return stitched_image, homography_matrix_list

    def detect_and_compute(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def warp_and_stitch(self, base_image, new_image, homography_matrix):
        height, width = new_image.shape[:2]
        corners = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)

        all_corners = np.concatenate((corners, transformed_corners), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel()) - 10
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel()) + 10

        translation_dist = [-x_min, -y_min]
        translation_matrix = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        stitched_image = cv2.warpPerspective(base_image, translation_matrix.dot(homography_matrix), (x_max-x_min, y_max-y_min))
        stitched_image[translation_dist[1]:translation_dist[1]+new_image.shape[0], translation_dist[0]:translation_dist[0]+new_image.shape[1]] = new_image

        return stitched_image

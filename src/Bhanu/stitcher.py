import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        images = [cv2.imread(img) for img in all_images]

        # Use the middle image as the anchor
        anchor_index = len(images) // 2
        f = (images[anchor_index].shape[1] / 2) / np.tan(np.pi / 5)

        # Perform cylindrical projection
        cylindrical_images = []
        for img in images:
            h, w = img.shape[:2]
            cylindrical_img = np.zeros_like(img)
            center_x, center_y = w // 2, h // 2

            for x in range(w):
                for y in range(h):
                    theta = (x - center_x) / f
                    h_ = (y - center_y) / f
                    X, Y, Z = np.sin(theta), h_, np.cos(theta)
                    x_img, y_img = int(f * X / Z + center_x), int(f * Y / Z + center_y)

                    if 0 <= x_img < w and 0 <= y_img < h:
                        cylindrical_img[y, x] = img[y_img, x_img]

            cylindrical_images.append(cylindrical_img)

        anchor_image = cylindrical_images[anchor_index]
        H_matrices = [None] * len(cylindrical_images)
        H_matrices[anchor_index] = np.eye(3)

        # Homography calculation and left-side warping
        for i in range(anchor_index - 1, -1, -1):
            detector = cv2.SIFT_create()
            keypoints1, descriptors1 = detector.detectAndCompute(cylindrical_images[i], None)
            keypoints2, descriptors2 = detector.detectAndCompute(cylindrical_images[i + 1], None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) < 4:
                print(f"Not enough matches found for images {i} and {i + 1}.")
                continue

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            optimal_H, inliers = self.RANSAC(src_pts, dst_pts, iterations=5000)
            H_matrices[i] = optimal_H @ H_matrices[i + 1]
            print(f"Left homography for image {i} calculated.")

        # Homography calculation and right-side warping
        for i in range(anchor_index + 1, len(cylindrical_images)):
            detector = cv2.SIFT_create()
            keypoints1, descriptors1 = detector.detectAndCompute(cylindrical_images[i - 1], None)
            keypoints2, descriptors2 = detector.detectAndCompute(cylindrical_images[i], None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) < 4:
                print(f"Not enough matches found for images {i - 1} and {i}.")
                continue

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            optimal_H, inliers = self.RANSAC(src_pts, dst_pts, iterations=5000)
            H_matrices[i] = np.linalg.inv(optimal_H) @ H_matrices[i - 1]
            print(f"Right homography for image {i} calculated.")

        # Calculate output size for the panorama
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for H in H_matrices:
            corners = np.array([[0, 0, 1], [anchor_image.shape[1], 0, 1],
                                [anchor_image.shape[1], anchor_image.shape[0], 1], [0, anchor_image.shape[0], 1]])
            transformed_corners = H @ corners.T
            transformed_corners /= transformed_corners[2, :]

            min_x = min(min_x, transformed_corners[0, :].min())
            max_x = max(max_x, transformed_corners[0, :].max())
            min_y = min(min_y, transformed_corners[1, :].min())
            max_y = max(max_y, transformed_corners[1, :].max())

        width = int(max_x - min_x)
        height = int(max_y - min_y)
        output_size = (width, height)

        panorama = cv2.warpPerspective(anchor_image, H_matrices[anchor_index], output_size)

        # Blend images into the panorama
        for i in range(len(cylindrical_images)):
            if i != anchor_index:
                warped_image = cv2.warpPerspective(cylindrical_images[i], H_matrices[i], output_size)
                mask = (warped_image > 0).astype(np.float32)
                panorama = (mask * warped_image + (1 - mask) * panorama).astype(panorama.dtype)
                print(f"Warped image {i} shape: {warped_image.shape}")

        # Crop black borders
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            panorama = panorama[y:y+h, x:x+w]

        return panorama, H_matrices

    def homography_matrix(self, src_pts, dst_pts): 
        A = []
        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0], src_pts[i][1]
            x2, y2 = dst_pts[i][0], dst_pts[i][1]
            z1 = z2 = 1
            A_i = [
                [z2 * x1, z2 * y1, z2 * z1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * z1],
                [0, 0, 0, -z2 * x1, -z2 * y1, -z2 * z1, y2 * x1, y2 * y1, y2 * z1]
            ]
            A.extend(A_i)

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        return H / H[2, 2]  

    def RANSAC(self, source_points, destination_points, iterations=1000, error_tolerance=1):
        tolerance_threshold = np.sqrt(5.99) * error_tolerance
        optimal_H = None
        optimal_inliers = []
        highest_inlier_count = 0

        for i in range(iterations):
            chosen_indices = np.random.choice(source_points.shape[0], 4, replace=False)
            sampled_source = source_points[chosen_indices]
            sampled_destination = destination_points[chosen_indices]

            H_candidate = self.homography_matrix(sampled_source, sampled_destination)

            source_homogeneous = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
            transformed_points = np.dot(H_candidate, source_homogeneous.T).T
            transformed_points /= transformed_points[:, 2].reshape(-1, 1)

            distance_errors = np.linalg.norm(destination_points - transformed_points[:, :2], axis=1)

            current_inliers = np.where(distance_errors < tolerance_threshold)[0]

            if len(current_inliers) > highest_inlier_count:
                highest_inlier_count = len(current_inliers)
                optimal_H = H_candidate
                optimal_inliers = current_inliers

        if len(optimal_inliers) > 4:
            optimal_H = self.homography_matrix(source_points[optimal_inliers], destination_points[optimal_inliers])

        return optimal_H, optimal_inliers

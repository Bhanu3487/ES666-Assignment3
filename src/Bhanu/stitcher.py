import cv2
import numpy as np
import glob
import os

class PanoramaStitcher:
    def make_panorama_for_images_in(self, path):
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
            keypoints1, descriptors1 = cv2.SIFT_create().detectAndCompute(cylindrical_images[i], None)
            keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(cylindrical_images[i + 1], None)

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

        panorama = self.warp_perspective(anchor_image, H_matrices[anchor_index], output_size)

        # Blend images into the panorama
        for i in range(len(cylindrical_images)):
            if i != anchor_index:
                warped_image = self.warp_perspective(cylindrical_images[i], H_matrices[i], output_size)
                mask = (warped_image > 0).astype(np.float32)
                panorama = (mask * warped_image + (1 - mask) * panorama).astype(panorama.dtype)
                print(f"Warped image {i} shape: {warped_image.shape}")

        # Crop black borders
        gray = np.dot(panorama[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        thresh = (gray > 1).astype(np.uint8) * 255
        
        contours = []
        height, width = thresh.shape
        visited = np.zeros_like(thresh, dtype=bool)

        for y in range(height):
            for x in range(width):
                if thresh[y, x] > 0 and not visited[y, x]:
                    # Found a new contour
                    contour = []
                    self.trace_contour(thresh, visited, x, y, contour)
                    contours.append(contour)

        if contours:
            x, y, w, h = self.bounding_rect(contours[0])
            panorama = panorama[y:y+h, x:x+w]

        return panorama, H_matrices

    def warp_perspective(self, image, H, output_size):
        # Get the dimensions of the input image
        h, w = image.shape[:2]

        # Prepare the output image
        output_image = np.zeros((output_size[1], output_size[0], 3), dtype=image.dtype)

        # Generate the inverse homography matrix
        H_inv = np.linalg.inv(H)

        # Iterate over every pixel in the output image
        for y in range(output_size[1]):
            for x in range(output_size[0]):
                # Create a homogeneous coordinate for the pixel
                pixel = np.array([x, y, 1])

                # Transform the pixel coordinate using the inverse homography matrix
                transformed_pixel = H_inv @ pixel
                transformed_pixel /= transformed_pixel[2]  # Normalize

                src_x, src_y = int(transformed_pixel[0]), int(transformed_pixel[1])

                # Check if the transformed pixel coordinates are within bounds of the input image
                if 0 <= src_x < w and 0 <= src_y < h:
                    output_image[y, x] = image[src_y, src_x]

        return output_image

    def trace_contour(self, binary_image, visited, start_x, start_y, contour):
        # Trace a contour starting from (start_x, start_y)
        height, width = binary_image.shape
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            contour.append((x, y))

            # Check 8-connectivity
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] > 0 and not visited[ny, nx]:
                    stack.append((nx, ny))

    def bounding_rect(self, contour):
        # Calculate the bounding rectangle of a contour
        x_coords = [p[0] for p in contour]
        y_coords = [p[1] for p in contour]
        x = min(x_coords)
        y = min(y_coords)
        w = max(x_coords) - x
        h = max(y_coords) - y
        return x, y, w, h

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

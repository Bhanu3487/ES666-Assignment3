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
        f = (images[anchor_index].shape[1] / 2) / np.tan(np.pi / 9)

        # Perform cylindrical projection
        cylindrical_images = self.create_cylindrical_images(images, f)

        anchor_image = cylindrical_images[anchor_index]
        H_matrices = [None] * len(cylindrical_images)
        H_matrices[anchor_index] = np.eye(3)

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

            optimal_H, inliers = self.RANSAC(src_pts, dst_pts, iters=5000)
            H_matrices[i] = np.linalg.inv(optimal_H) @ H_matrices[i - 1]
            print(f"Right homography for image {i} calculated.")

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

            optimal_H, inliers = self.RANSAC(src_pts, dst_pts, iters=5000)
            H_matrices[i] = optimal_H @ H_matrices[i + 1]
            print(f"Left homography for image {i} calculated.")

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

        def create_cylindrical_images(self, images, f):
        cylindrical_images = []
        for img in images:
            h, w = img.shape[:2]
            cylindrical_img = np.zeros_like(img)
            xc, yc = w // 2, h // 2  # Center coordinates

            for x in range(w):
                for y in range(h):
                    # projection onto cylinder
                    theta = (x - xc) / f
                    h_ = (y - yc) / f
                    #unroll
                    X, Y, Z = np.sin(theta), h_, np.cos(theta)

                    # Calculate the new coordinates
                    x_dash = int(f * X / Z + xc)
                    y_dash = int(f * Y / Z + yc)

                    if 0 <= x_dash < w and 0 <= y_dash < h:
                        cylindrical_img[y, x] = img[y_dash, x_dash]

            cylindrical_images.append(cylindrical_img)
        return cylindrical_images
       

    def warp_perspective(self, image, H, output_size):
        # Get dimensions of the input image and prepare the output image
        h, w = image.shape[:2]
        output_image = np.zeros((output_size[1], output_size[0], 3), dtype=image.dtype)

        # Generate the inverse homography matrix (wkt H is full rank)
        H_inv = np.linalg.inv(H)

        # Create a mesh grid of output coordinates; x_indices holds the x-coordinates and y_indices holds the y-coordinates for all pixels in the output image.
        x_indices, y_indices = np.meshgrid(np.arange(output_size[0]), np.arange(output_size[1]))
        pixel_coords = np.vstack((x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices.ravel()))) # of form [x,y,1]

        # Apply the perspective transformation using the inverse homography matrix.
        transformed_pixels = H_inv @ pixel_coords
        transformed_pixels /= transformed_pixels[2]  # non homogeneous coordiantes

        src_x = transformed_pixels[0].astype(int)
        src_y = transformed_pixels[1].astype(int)

        # Check bounds and assign pixels
        valid_mask = (0 <= src_x) & (src_x < w) & (0 <= src_y) & (src_y < h)
        output_image[y_indices.ravel()[valid_mask], x_indices.ravel()[valid_mask]] = image[src_y[valid_mask], src_x[valid_mask]]

        return output_image


    def trace_contour(self, binary_image, visited, start_x, start_y, contour):
        # Trace a contour starting from (start_x, start_y) 
        height, width = binary_image.shape
        stack = [(start_x, start_y)]
        
        # DFS to identify and trace the boundaries of objects (in binary images)
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            contour.append((x, y))

            # Check all 8 neighbours
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] > 0 and not visited[ny, nx]:
                    stack.append((nx, ny))
    
    def bounding_rect(self, contour):
        # Calculate the bounding rectangle of a contour: list of points (e.g., [(x1, y1), (x2, y2), ...])
        contour = np.array(contour) 
        x_min, y_min = contour.min(axis=0)
        x_max, y_max = contour.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        return x_min, y_min, w, h

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

    def RANSAC(self, src_pts, dst_pts, iters=500, tol=3):
        # sq(5.99) for 95% confidence; tolerance can to set to 1-3
        threshold = np.sqrt(5.99) * tol  
        best_H = None
        best_inliers = []
        max_inliers = 0

        for _ in range(iters):
            # Randomly pick 4 points for homography estimation since we require 4 points correspondances to estimate H matrix
            idx = np.random.choice(src_pts.shape[0], 4, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]

            # Estimate homography from 4-point sample
            H = self.homography_matrix(src_sample, dst_sample)

            # Transform all source points using candidate homography
            src_homog = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
            transformed = np.dot(H, src_homog.T).T
            # Normalize to non homogeneous coords
            transformed /= transformed[:, 2].reshape(-1, 1)  

            # Calculate distance errors between transformed source and destination points
            errors = np.linalg.norm(dst_pts - transformed[:, :2], axis=1)

            # Identify inliers based on error threshold
            inliers = np.where(errors < threshold)[0]

            # Update best homography if more inliers are found
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_H = H
                best_inliers = inliers

        # Refine homography using all inliers if there are enough
        if len(best_inliers) > 4:
            best_H = self.homography_matrix(src_pts[best_inliers], dst_pts[best_inliers])

        return best_H, best_inliers

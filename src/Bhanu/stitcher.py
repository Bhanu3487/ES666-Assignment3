# import cv2
# import numpy as np
# import glob
# import os

# class PanaromaStitcher:      
#     def make_panaroma_for_images_in(self, path):
#         # Load and prepare images and calculate focal length
#         imgs, middle_index = self.load_and_prepare_images(path)
#         focal_length = self.calculate_focal_length(imgs[middle_index])

#         # Perform cylindrical projection on images
#         cylindrical_imgs = self.create_cylindrical_images(imgs, focal_length)

#         # Calculate homographies and create panorama canvas
#         H_matrices = self.calculate_homographies(cylindrical_imgs, middle_index)
#         output_size = self.calc_output_size(cylindrical_imgs[middle_index], H_matrices)

#         # Stitch images onto the panorama canvas
#         panorama = self.stitch_images(cylindrical_imgs, H_matrices, middle_index, output_size)
#         panorama = self.blend(cylindrical_imgs, middle_index, H_matrices, panorama, output_size)
#         panorama = self.crop_black_borders(panorama) # crop the black borders

#         return panorama, H_matrices
    
#     def blend(self, cylindrical_images, middle_index, H_matrices, panorama, output_size):
#         for i in range(len(cylindrical_images)):
#             if i != middle_index:
#                 warped_image = self.warp_perspective(cylindrical_images[i], H_matrices[i], output_size)
#                 mask = (warped_image > 0).astype(np.float32)
#                 panorama = (mask * warped_image + (1 - mask) * panorama).astype(panorama.dtype)
#                 print(f"Warped image {i} shape: {warped_image.shape}")
#         return panorama

#     def load_and_prepare_images(self, path):
#         all_imgs = sorted(glob.glob(path + os.sep + '*'))
#         imgs = [cv2.imread(img) for img in all_imgs]
#         middle_index = len(imgs) // 2
#         return imgs, middle_index

#     def calculate_focal_length(self, anchor_img):
#         # Using a 75-degree field of view
#         return (anchor_img.shape[1] / 2) / np.tan(np.pi / 10)

#     def calculate_homographies(self, cylindrical_imgs, middle_index):
#         H_matrices = [None] * len(cylindrical_imgs)
#         H_matrices[middle_index] = np.eye(3)  # Identity matrix for anchor image

#         # Calculate left-side homographies
#         self.calculate_left_homographies(cylindrical_imgs, middle_index, H_matrices)
        
#         # Calculate right-side homographies
#         self.calculate_right_homographies(cylindrical_imgs, middle_index, H_matrices)

#         return H_matrices

#     def calculate_right_homographies(self, cylindrical_imgs, middle_index, H_matrices):
#         for i in range(middle_index + 1, len(cylindrical_imgs)):
#             kp1, desc1 = self.detect_features(cylindrical_imgs[i - 1])
#             kp2, desc2 = self.detect_features(cylindrical_imgs[i])

#             good_matches = self.match_features(desc1, desc2)

#             if len(good_matches) < 4:
#                 print(f"Not enough matches found for images {i - 1} and {i}.")
#                 continue

#             src_pts, dst_pts = self.get_matched_points(kp1, kp2, good_matches)
#             optimal_H, _ = self.RANSAC(src_pts, dst_pts, iters=500)
#             H_matrices[i] = np.linalg.inv(optimal_H) @ H_matrices[i - 1]
#             print(f"Right homography for image {i} calculated.")

#     def calculate_left_homographies(self, cylindrical_imgs, middle_index, H_matrices):
#         for i in range(middle_index - 1, -1, -1):
#             kp1, desc1 = self.detect_features(cylindrical_imgs[i])
#             kp2, desc2 = self.detect_features(cylindrical_imgs[i + 1])

#             good_matches = self.match_features(desc1, desc2)

#             if len(good_matches) < 4:
#                 print(f"Not enough matches found for images {i} and {i + 1}.")
#                 continue

#             src_pts, dst_pts = self.get_matched_points(kp1, kp2, good_matches)
#             optimal_H, _ = self.RANSAC(src_pts, dst_pts, iters=5000)
#             H_matrices[i] = optimal_H @ H_matrices[i + 1]
#             print(f"Left homography for image {i} calculated.")

#     def detect_features(self, img):
#         detector = cv2.SIFT_create(nfeatures=500)
#         return detector.detectAndCompute(img, None)

#     def match_features(self, desc1, desc2):
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(desc1, desc2, k=2)
#         return [m for m, n in matches if m.distance < 0.75 * n.distance]

#     def get_matched_points(self, kp1, kp2, good_matches):
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
#         return src_pts, dst_pts

#     def stitch_images(self, cylindrical_imgs, H_matrices, middle_index, output_size):
#         anchor_img = cylindrical_imgs[middle_index]
#         panorama = self.warp_perspective(anchor_img, H_matrices[middle_index], output_size)

#         for i in range(len(cylindrical_imgs)):
#             if i != middle_index:
#                 warped_img = self.warp_perspective(cylindrical_imgs[i], H_matrices[i], output_size)
#                 mask = (warped_img > 0).astype(np.float32)
#                 panorama = (mask * warped_img + (1 - mask) * panorama).astype(panorama.dtype)
#                 print(f"Warped image {i} added to panorama.")

#         return panorama

#     def crop_black_borders(self, panorama):
#         gray = np.dot(panorama[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
#         thresh = (gray > 1).astype(np.uint8) * 255

#         contours = []
#         height, width = thresh.shape
#         visited = np.zeros_like(thresh, dtype=bool)

#         for y in range(height):
#             for x in range(width):
#                 if thresh[y, x] > 0 and not visited[y, x]:
#                     contour = []
#                     self.trace_contour(thresh, visited, x, y, contour)
#                     contours.append(contour)

#         if contours:
#             x, y, w, h = self.bounding_rect(contours[0])
#             panorama = panorama[y:y + h, x:x + w]

#         return panorama
   
#     def create_cylindrical_images(self, images, f):
#         cylindrical_images = []
#         for img in images:
#             h, w = img.shape[:2]
#             cylindrical_img = np.zeros_like(img)
#             xc, yc = w // 2, h // 2  # Center coordinates

#             for x in range(w):
#                 for y in range(h):
#                     # projection onto cylinder
#                     theta = (x - xc) / f
#                     h_ = (y - yc) / f
#                     #unroll
#                     X, Y, Z = np.sin(theta), h_, np.cos(theta)

#                     # Calculate the new coordinates
#                     x_dash = int(f * X / Z + xc)
#                     y_dash = int(f * Y / Z + yc)

#                     if 0 <= x_dash < w and 0 <= y_dash < h:
#                         cylindrical_img[y, x] = img[y_dash, x_dash]

#             cylindrical_images.append(cylindrical_img)
#         return cylindrical_images

#     def calc_output_size(self, anchor_image, H_matrices):
#         min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
#         for H in H_matrices:
#             corners = np.array([[0, 0, 1], [anchor_image.shape[1], 0, 1],
#                                 [anchor_image.shape[1], anchor_image.shape[0], 1], [0, anchor_image.shape[0], 1]])
#             transformed_corners = H @ corners.T
#             transformed_corners /= transformed_corners[2, :]

#             min_x = min(min_x, transformed_corners[0, :].min())
#             max_x = max(max_x, transformed_corners[0, :].max())
#             min_y = min(min_y, transformed_corners[1, :].min())
#             max_y = max(max_y, transformed_corners[1, :].max())

#         width = int(max_x - min_x)
#         height = int(max_y - min_y)
#         return (width, height)
    
#     def warp_perspective(self, image, H, output_size):
#         # Get dimensions of the input image and prepare the output image
#         h, w = image.shape[:2]
#         output_image = np.zeros((output_size[1], output_size[0], 3), dtype=image.dtype)

#         # Generate the inverse homography matrix (wkt H is full rank)
#         H_inv = np.linalg.inv(H)

#         # Create a mesh grid of output coordinates; x_indices holds the x-coordinates and y_indices holds the y-coordinates for all pixels in the output image.
#         x_indices, y_indices = np.meshgrid(np.arange(output_size[0]), np.arange(output_size[1]))
#         pixel_coords = np.vstack((x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices.ravel()))) # of form [x,y,1]

#         # Apply the perspective transformation using the inverse homography matrix.
#         transformed_pixels = H_inv @ pixel_coords
#         transformed_pixels /= transformed_pixels[2]  # non homogeneous coordiantes

#         src_x = transformed_pixels[0].astype(int)
#         src_y = transformed_pixels[1].astype(int)

#         # Check bounds and assign pixels
#         valid_mask = (0 <= src_x) & (src_x < w) & (0 <= src_y) & (src_y < h)
#         output_image[y_indices.ravel()[valid_mask], x_indices.ravel()[valid_mask]] = image[src_y[valid_mask], src_x[valid_mask]]

#         return output_image

#     def trace_contour(self, binary_image, visited, start_x, start_y, contour):
#         # Trace a contour starting from (start_x, start_y) 
#         height, width = binary_image.shape
#         stack = [(start_x, start_y)]
        
#         # DFS to identify and trace the boundaries of objects (in binary images)
#         while stack:
#             x, y = stack.pop()
#             if visited[y, x]:
#                 continue
            
#             visited[y, x] = True
#             contour.append((x, y))

#             # Check all 8 neighbours
#             for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] > 0 and not visited[ny, nx]:
#                     stack.append((nx, ny))
    
#     def bounding_rect(self, contour):
#         # Calculate the bounding rectangle of a contour: list of points (e.g., [(x1, y1), (x2, y2), ...])
#         contour = np.array(contour) 
#         x_min, y_min = contour.min(axis=0)
#         x_max, y_max = contour.max(axis=0)
#         w, h = x_max - x_min, y_max - y_min
#         return x_min, y_min, w, h

#     def homography_matrix(self, src_pts, dst_pts): 
#         A = []
#         for i in range(len(src_pts)):
#             x1, y1 = src_pts[i][0], src_pts[i][1]
#             x2, y2 = dst_pts[i][0], dst_pts[i][1]
#             z1 = z2 = 1
#             A_i = [
#                 [z2 * x1, z2 * y1, z2 * z1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * z1],
#                 [0, 0, 0, -z2 * x1, -z2 * y1, -z2 * z1, y2 * x1, y2 * y1, y2 * z1]
#             ]
#             A.extend(A_i)

#         A = np.array(A)
#         _, _, V = np.linalg.svd(A)
#         H = V[-1].reshape((3, 3))

#         return H / H[2, 2]  
    
#     def RANSAC(self, src_pts, dst_pts, iters=500, tol=3):
#         # sq(5.99) for 95% confidence; tolerance can to set to 1-3
#         threshold = np.sqrt(5.99) * tol  
#         best_H = None
#         best_inliers = []
#         max_inliers = 0

#         for _ in range(iters):
#             # Randomly pick 4 points for homography estimation since we require 4 points correspondances to estimate H matrix
#             idx = np.random.choice(src_pts.shape[0], 4, replace=False)
#             src_sample = src_pts[idx]
#             dst_sample = dst_pts[idx]

#             # Estimate homography from 4-point sample
#             H = self.homography_matrix(src_sample, dst_sample)

#             # Transform all source points using candidate homography
#             src_homog = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
#             transformed = np.dot(H, src_homog.T).T
#             # Normalize to non homogeneous coords
#             transformed /= transformed[:, 2].reshape(-1, 1)  

#             # Calculate distance errors between transformed source and destination points
#             errors = np.linalg.norm(dst_pts - transformed[:, :2], axis=1)

#             # Identify inliers based on error threshold
#             inliers = np.where(errors < threshold)[0]

#             # Update best homography if more inliers are found
#             if len(inliers) > max_inliers:
#                 max_inliers = len(inliers)
#                 best_H = H
#                 best_inliers = inliers

#         # Refine homography using all inliers if there are enough
#         if len(best_inliers) > 4:
#             best_H = self.homography_matrix(src_pts[best_inliers], dst_pts[best_inliers])

#         return best_H, best_inliers


import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:     
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

    # def warp_perspective(self, image, H, output_size):
    #     # Get the dimensions of the input image and prepare the output image
    #     h, w = image.shape[:2]
    #     output_image = np.zeros((output_size[1], output_size[0], 3), dtype=image.dtype)

    #     # Generate the inverse homography matrix (wkt H is full rank)
    #     H_inv = np.linalg.inv(H)

    #     # Iterate over every pixel in the output image
    #     for y in range(output_size[1]):
    #         for x in range(output_size[0]):
    #             # Create a homogeneous coordinate for the pixel
    #             pixel = np.array([x, y, 1])

    #             # Transform the pixel coordinate using the inverse homography matrix
    #             transformed_pixel = H_inv @ pixel
    #             transformed_pixel /= transformed_pixel[2]  # Normalize

    #             src_x, src_y = int(transformed_pixel[0]), int(transformed_pixel[1])

    #             # Check if the transformed pixel coordinates are within bounds of the input image
    #             if 0 <= src_x < w and 0 <= src_y < h:
    #                 output_image[y, x] = image[src_y, src_x]

    #     return output_image

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

    # def bounding_rect(self, contour):
    #     # Calculate the bounding rectangle of a contour
    #     x_coords = [p[0] for p in contour]
    #     y_coords = [p[1] for p in contour]
    #     x = min(x_coords)
    #     y = min(y_coords)
    #     w = max(x_coords) - x
    #     h = max(y_coords) - y
    #     return x, y, w, h
    
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

    # def RANSAC(self, source_points, destination_points, v=1000, error_tolerance=1):
    #     tolerance_threshold = np.sqrt(5.99) * error_tolerance
    #     optimal_H = None
    #     optimal_inliers = []
    #     highest_inlier_count = 0

    #     for i in range(iters):
    #         chosen_indices = np.random.choice(source_points.shape[0], 4, replace=False)
    #         sampled_source = source_points[chosen_indices]
    #         sampled_destination = destination_points[chosen_indices]

    #         H_candidate = self.homography_matrix(sampled_source, sampled_destination)

    #         source_homogeneous = np.hstack([source_points, np.ones((source_points.shape[0], 1))])
    #         transformed_points = np.dot(H_candidate, source_homogeneous.T).T
    #         transformed_points /= transformed_points[:, 2].reshape(-1, 1)

    #         distance_errors = np.linalg.norm(destination_points - transformed_points[:, :2], axis=1)

    #         current_inliers = np.where(distance_errors < tolerance_threshold)[0]

    #         if len(current_inliers) > highest_inlier_count:
    #             highest_inlier_count = len(current_inliers)
    #             optimal_H = H_candidate
    #             optimal_inliers = current_inliers

    #     if len(optimal_inliers) > 4:
    #         optimal_H = self.homography_matrix(source_points[optimal_inliers], destination_points[optimal_inliers])

    #     return optimal_H, optimal_inliers
    
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

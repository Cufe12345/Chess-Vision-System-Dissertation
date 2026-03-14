import math

import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import os
import tensorflow as tf

#todo -clean up code and add comments

def getSquaresFromImage(image_path,debug=False,colour=False,farChessTable=False):
        model = loadBinaryModel("binary_classification_model.h5", "binary_classification_weights.h5")
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        margin_x = int(0.05 * w)  # 5% of width
        margin_y = int(0.05 * h)  # 5% of height

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        
        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Preprocessed Grayscale Image")
            plt.imshow(grey, cmap='gray')
            plt.axis('off')
            plt.show()
        # grey = np.float32(grey)

        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Blurred Image")
            plt.imshow(blur, cmap='gray')
            plt.axis('off')
            plt.show()
        

        temp = np.float32(blur)

        #show image
        # cv2.imshow("Blurred Image", blur)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        test = cv2.cornerHarris(temp,4,3,0.05)
        test = cv2.dilate(test, None)

        # Show Harris response
        # plt.figure(figsize=(6,6))
        # plt.title("Harris Corner Response")
        # plt.imshow(test, cmap='hot')
        # plt.axis('off')
        # plt.show()

        thresh = 0.025 * test.max()
        corners = np.argwhere(test > thresh)

        corners = np.flip(corners, axis=1)  # Flip to (x, y) format

        # Method 2
        # filtered_corners = []
        # for x, y in corners:
        #     if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
        #         filtered_corners.append([x, y])

        # filtered_corners = np.array(filtered_corners)
        # print(f"{len(filtered_corners)} corners remain after edge filtering")

        # img_corners = img.copy()
        # for x, y in filtered_corners:
        #     cv2.circle(img_corners, (x, y), 3, (0, 0, 255), -1)

        img_corners = img.copy()
        for x, y in corners:
            cv2.circle(img_corners, (x, y), 3, (0, 0, 255), -1)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Detected Corners")
            plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        responses = test[corners[:,1], corners[:,0]]
        idx = np.argsort(responses)[::-1]  # strongest first
        points = corners[idx]

        min_dist = 10  # pixels (tune: 8–15 works well)

        # filtered = []

        # for p in points:
        #     if all(np.linalg.norm(p - np.array(q)) > min_dist for q in filtered):
        #         filtered.append(p)

        # filtered = np.array(filtered)

        # points = filtered
        # center = points.mean(axis=0)
        # cx, cy = int(center[0]), int(center[1])

        # distances = np.linalg.norm(points - center, axis=1)

        # mean_dist = np.mean(distances)
        # std_dist = np.std(distances)

        # k = 1.5  # start with 1.5–2.0
        # mask = distances < (mean_dist + k * std_dist)

        # filtered_points = points[mask]
        filtered = []

        for p in points:
            if not filtered:
                filtered.append(p)
                continue

            dists = np.linalg.norm(np.array(filtered) - p, axis=1)
            if np.all(dists > min_dist):
                filtered.append(p)

        filtered = np.array(filtered)

        points = filtered
        center = points.mean(axis=0)
        cx, cy = int(center[0]), int(center[1])

        distances = np.linalg.norm(points - center, axis=1)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        k = 1.5  # start with 1.5–2.0
        mask = distances < (mean_dist + k * std_dist)

        filtered_points = points[mask]

        if debug:
            #show filtered points
            img_filtered = img.copy()
            for x, y in filtered_points:
                cv2.circle(img_filtered, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(img_filtered, (cx, cy), 8, (0, 255, 255), -1)   # yellow
            cv2.putText(img_filtered, "Mean", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            plt.figure(figsize=(6,6))
            plt.title("Filtered Corners")
            plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        # filtered_points = filtered

        # img_before = img.copy()
        # for x, y in points:
        #     cv2.circle(img_before, (int(x), int(y)), 3, (0, 0, 255), -1)

        # # Draw mean point (before)
        # cv2.circle(img_before, (cx, cy), 8, (0, 255, 255), -1)   # yellow
        # cv2.putText(img_before, "Mean", (cx+10, cy),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)


        # img_after = img.copy()
        # for x, y in filtered_points:
        #     cv2.circle(img_after, (int(x), int(y)), 3, (0, 255, 0), -1)

        # # Draw mean point (after)
        # cv2.circle(img_after, (cx, cy), 8, (0, 255, 255), -1)
        # cv2.putText(img_after, "Mean", (cx+10, cy),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # if debug:
        #     plt.figure(figsize=(12,6))

        #     plt.subplot(1,2,1)
        #     plt.title("Before Center-Based Filtering")
        #     plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')

        #     plt.subplot(1,2,2)
        #     plt.title("After Center-Based Filtering")
        #     plt.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')

        #     plt.show()


        hull = cv2.convexHull(filtered_points.astype(np.int32))

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        roi = cv2.bitwise_and(img, img, mask=mask)

        if farChessTable:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=40,
                maxLineGap=20
        )
        #changed min line length from 200 to 20
        #Changed threshold from 100 to 80

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Canny Edge Map")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

            img_all_lines = img.copy()

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_all_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)

            plt.figure(figsize=(6,6))
            plt.title("All Hough Lines (Before Filtering)")
            plt.imshow(cv2.cvtColor(img_all_lines, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)

            if abs(angle) < np.pi / 12:            # near horizontal
                h_lines.append((x1,y1,x2,y2))
            elif abs(abs(angle) - np.pi/2) < np.pi / 12:  # near vertical
                v_lines.append((x1,y1,x2,y2))

        top = min(h_lines, key=lambda l: min(l[1], l[3]))
        bottom = max(h_lines, key=lambda l: max(l[1], l[3]))
        left = min(v_lines, key=lambda l: min(l[0], l[2]))
        right = max(v_lines, key=lambda l: max(l[0], l[2]))

        if debug:
            img_lines = img.copy()
            for l in [top, bottom, left, right]:
                cv2.line(img_lines, (l[0],l[1]), (l[2],l[3]), (0,255,0), 3)

            plt.figure(figsize=(6,6))
            plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        def intersect(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            A1 = y2 - y1
            B1 = x1 - x2
            C1 = A1 * x1 + B1 * y1

            A2 = y4 - y3
            B2 = x3 - x4
            C2 = A2 * x3 + B2 * y3

            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-6:
                return None  # parallel lines

            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return int(x), int(y)
        tl = intersect(top, left)
        tr = intersect(top, right)
        br = intersect(bottom, right)
        bl = intersect(bottom, left)

        board_corners = np.array([tl, tr, br, bl])

        if debug:
            img_square = img.copy()

            # Draw edges
            for i in range(4):
                p1 = tuple(board_corners[i])
                p2 = tuple(board_corners[(i+1) % 4])
                cv2.line(img_square, p1, p2, (0, 255, 0), 3)

            # Draw corners
            for x, y in board_corners:
                cv2.circle(img_square, (x, y), 6, (0, 0, 255), -1)

            plt.figure(figsize=(6,6))
            plt.title("Detected Chessboard Outline (Hough)")
            plt.imshow(cv2.cvtColor(img_square, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()



            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [board_corners.astype(np.int32)], 255)

            segmented = cv2.bitwise_and(img, img, mask=mask)

            plt.figure(figsize=(6,6))
            plt.title("Segmented Board (Masked)")
            plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        out_size = 800

        dst_pts = np.array([
            [0, 0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0, out_size - 1]
        ], dtype=np.float32)

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(img, H, (out_size, out_size))

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Warped Chessboard (Top-Down View)")
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)

        edges = cv2.Canny(gray_warped, 50, 150)


        grid_img = warped.copy()

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Edges on Warped Board")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        # Thicken edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Connect broken lines
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Edges on Warped Board 2")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=warped.shape[0] // 4,
            maxLineGap=30
        )

        if debug:
            all_lines_vis = warped.copy()

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(all_lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

            plt.figure(figsize=(6,6))
            plt.title("All Hough Lines (Warped Image)")
            plt.imshow(cv2.cvtColor(all_lines_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


        h_lines = []
        v_lines = []

        angle_thresh = np.pi / 10  # ~18 degrees

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)

            # Horizontal
            if abs(angle) < angle_thresh:
                h_lines.append((x1, y1, x2, y2))

            # Vertical
            elif abs(abs(angle) - np.pi / 2) < angle_thresh:
                v_lines.append((x1, y1, x2, y2))

        if debug:
            h_vis = warped.copy()
            for l in h_lines:
                cv2.line(h_vis, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Horizontal Line Candidates")
            plt.imshow(cv2.cvtColor(h_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            v_vis = warped.copy()
            for l in v_lines:
                cv2.line(v_vis, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Vertical Line Candidates")
            plt.imshow(cv2.cvtColor(v_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        def merge_similar_lines(lines, axis='x', dist_thresh=15):
            """
            axis='x' → vertical lines (merge by x)
            axis='y' → horizontal lines (merge by y)
            """
            if axis == 'x':
                key = lambda l: (l[0] + l[2]) / 2
            else:
                key = lambda l: (l[1] + l[3]) / 2

            lines = sorted(lines, key=key)
            merged = []

            for line in lines:
                if not merged:
                    merged.append(line)
                    continue

                prev = merged[-1]
                dist = abs(key(line) - key(prev))

                if dist < dist_thresh:
                    # keep longer line
                    len_new = np.hypot(line[2]-line[0], line[3]-line[1])
                    len_old = np.hypot(prev[2]-prev[0], prev[3]-prev[1])

                    if len_new > len_old:
                        merged[-1] = line
                else:
                    merged.append(line)

            return merged

        h_lines = merge_similar_lines(h_lines, axis='y', dist_thresh=15)
        v_lines = merge_similar_lines(v_lines, axis='x', dist_thresh=15)

        if debug:

            h_vis = warped.copy()
            for l in h_lines:
                cv2.line(h_vis, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Horizontal Line Candidates after")
            plt.imshow(cv2.cvtColor(h_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            v_vis = warped.copy()
            for l in v_lines:
                cv2.line(v_vis, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Vertical Line Candidates after")
            plt.imshow(cv2.cvtColor(v_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


        def extend_line(line, img_shape):
            x1, y1, x2, y2 = line
            h, w = img_shape[:2]

            if x1 == x2:  # vertical
                return (x1, 0, x1, h-1)

            if y1 == y2:  # horizontal
                return (0, y1, w-1, y1)

            # For completeness (diagonal case)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

            points = []

            # Left border (x = 0)
            y = b
            if 0 <= y < h:
                points.append((0, int(y)))

            # Right border (x = w-1)
            y = m * (w-1) + b
            if 0 <= y < h:
                points.append((w-1, int(y)))

            # Top border (y = 0)
            x = -b / m
            if 0 <= x < w:
                points.append((int(x), 0))

            # Bottom border (y = h-1)
            x = (h-1 - b) / m
            if 0 <= x < w:
                points.append((int(x), h-1))

            if len(points) >= 2:
                return (*points[0], *points[1])

            return line  # fallback

        h_lines_ext = [extend_line(l, warped.shape) for l in h_lines]
        v_lines_ext = [extend_line(l, warped.shape) for l in v_lines]

        if debug:
            h_vis = warped.copy()
            for l in h_lines_ext:
                cv2.line(h_vis, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Horizontal Line Candidates after Extension")
            plt.imshow(cv2.cvtColor(h_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            v_vis = warped.copy()
            for l in v_lines_ext:
                cv2.line(v_vis, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Vertical Line Candidates after Extension")
            plt.imshow(cv2.cvtColor(v_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


        def clip_vertical_line(line, y_top, y_bottom):
            """
            line: (x1,y1,x2,y2)
            y_top, y_bottom: vertical bounds
            Returns clipped line (x1, y_top, x2, y_bottom)
            """

            x1, y1, x2, y2 = line

            if y2 == y1:
                return line  # avoid division by zero

            m = (x2 - x1) / (y2 - y1)  # dx/dy since we're solving for x

            x_top = x1 + (y_top - y1) * m
            x_bottom = x1 + (y_bottom - y1) * m

            return (int(x_top), int(y_top), int(x_bottom), int(y_bottom))

        def clip_horizontal_line(line, x_left, x_right):
            """
            line: (x1,y1,x2,y2)
            x_left, x_right: horizontal bounds
            Returns clipped line (x_left, y1, x_right, y2)
            """
            x1, y1, x2, y2 = line

            if x2 == x1:
                return line

            m = (y2 - y1) / (x2 - x1)

            y_left = y1 + (x_left - x1) * m
            y_right = y1 + (x_right - x1) * m

            return (int(x_left), int(y_left), int(x_right), int(y_right))

        y_top = min((l[1] + l[3])/2 for l in h_lines_ext)
        y_bottom = max((l[1] + l[3])/2 for l in h_lines_ext)

        x_left = min((l[0] + l[2])/2 for l in v_lines_ext)
        x_right = max((l[0] + l[2])/2 for l in v_lines_ext)

        print(f"Clipping bounds: y_top={y_top}, y_bottom={y_bottom}, x_left={x_left}, x_right={x_right}")

        v_lines_clipped = [clip_vertical_line(l, y_top, y_bottom) for l in v_lines_ext]
        h_lines_clipped = [clip_horizontal_line(l, x_left, x_right) for l in h_lines_ext]

        if debug:
            h_vis = warped.copy()
            for l in h_lines_clipped:
                x1, y1, x2, y2 = map(int, l)
                cv2.line(h_vis, (x1, y1), (x2, y2), (255,0,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Horizontal Line Candidates after Extension and clipping")
            plt.imshow(cv2.cvtColor(h_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            v_vis = warped.copy()
            for l in v_lines_clipped:
                x1, y1, x2, y2 = map(int, l)
                cv2.line(v_vis, (x1, y1), (x2, y2), (0,255,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Vertical Line Candidates after Extension and clipping")
            plt.imshow(cv2.cvtColor(v_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


        h_lines = h_lines_clipped
        v_lines = v_lines_clipped

        h_lines = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_lines = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        def line_angle(l):
            x1,y1,x2,y2 = l
            return np.arctan2(y2-y1, x2-x1)

        def select_best_9_spacing(lines, axis='x', img_size=None, angle_tol_deg=20):
            """
            Select the best 9 lines based on spacing uniformity.
            lines: list of lines (x1,y1,x2,y2)
            axis: 'x' for vertical, 'y' for horizontal
            img_size: size of image along the axis
            """
            if len(lines) < 9:
                return None

            h_img, w_img = warped.shape[:2]

            if axis == 'x':
                slices = [0.1*h_img, 0.5*h_img, 0.9*h_img]
            else:
                slices = [0.1*w_img, 0.5*w_img, 0.9*w_img]

            positions = []
            angles = []
            lines_valid = []

            for l in lines:

                x1, y1, x2, y2 = l
                coords = []

                if axis == 'x':

                    if abs(y2 - y1) < 1e-6:
                        continue

                    for y in slices:
                        t = (y - y1) / (y2 - y1)

                        if 0 <= t <= 1:
                            x = x1 + t * (x2 - x1)
                            coords.append(x)

                else:

                    if abs(x2 - x1) < 1e-6:
                        continue

                    for x in slices:
                        t = (x - x1) / (x2 - x1)

                        if 0 <= t <= 1:
                            y = y1 + t * (y2 - y1)
                            coords.append(y)

                if not coords:
                    continue

                positions.append(np.mean(coords))
                angles.append(math.atan2(y2 - y1, x2 - x1))
                lines_valid.append(l)

            if len(lines_valid) < 9:
                return None

            positions = np.array(positions)
            angles = np.array(angles)

            # --- ANGLE PRUNING ---
            median_angle = np.median(angles)
            angle_tol = np.deg2rad(angle_tol_deg)

            mask = np.abs(angles - median_angle) < angle_tol

            positions = positions[mask]
            angles = angles[mask]
            lines_valid = [l for l, m in zip(lines_valid, mask) if m]

            if len(lines_valid) < 9:
                return None

            # sort by position
            order = np.argsort(positions)

            positions = positions[order]
            angles = angles[order]
            lines_sorted = [lines_valid[i] for i in order]

            best_score = np.inf
            best_group = None

            for combo_indices in itertools.combinations(range(len(lines_sorted)), 9):

                combo_positions = positions[list(combo_indices)]
                combo_angles = angles[list(combo_indices)]

                diffs = np.diff(combo_positions)

                spacing_mean = np.mean(diffs)
                spacing_std = np.std(diffs)

                total_span = combo_positions[-1] - combo_positions[0]

                if spacing_mean < 10:
                    continue

                if img_size is not None and total_span < 0.3 * img_size:
                    continue

                angle_std = np.std(combo_angles)

                score = spacing_std + 200 * angle_std

                if score < best_score:
                    best_score = score
                    best_group = [lines_sorted[i] for i in combo_indices]

            return best_group

        v_grid = select_best_9_spacing(v_lines, axis='x', img_size=warped.shape[1])
        h_grid = select_best_9_spacing(h_lines, axis='y', img_size=warped.shape[0])

        if debug:
            grid_vis = warped.copy()

            for l in v_grid + h_grid:
                x1,y1,x2,y2 = l
                x1, y1, x2, y2 = map(int, l)
                cv2.line(grid_vis, (x1,y1), (x2,y2), (0,255,0), 2)

            plt.figure(figsize=(6,6))
            plt.title("Selected 9x9 Grid Lines")
            plt.imshow(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        grid_points = []

        for h in h_grid:
            row = []
            for v in v_grid:
                row.append(intersect(h, v))
            grid_points.append(row)

        grid_points = np.array(grid_points)

        squares = []

        for i in range(8):  # rows
            for j in range(8):  # columns
                tl = grid_points[i][j]
                tr = grid_points[i][j+1]
                br = grid_points[i+1][j+1]
                bl = grid_points[i+1][j]
                squares.append(np.array([tl, tr, br, bl]))

        vis = warped.copy()

        for sq in squares:
            pts = sq.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        square_images = []

        for sq in squares:
            sq_t = sq.astype(int)

            xs = sq_t[:, 0]
            ys = sq_t[:, 1]

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            square_img = warped[y_min:y_max, x_min:x_max]

            # # Optional: make uniform size for all squares
            # square_img = cv2.resize(square_img, (64, 64))  # 64x64 px for consistency
            # plt.figure(figsize=(4,4))
            # plt.title("Initial Square")
            # plt.imshow(cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()
            square_img =  expand_square_if_not_empty(model,warped, sq, margin=50,colour=colour,farChessTable=farChessTable)

            # plt.figure(figsize=(4,4))
            # plt.title("Expanded Square")
            # plt.imshow(cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.show()
            square_images.append(square_img)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Detected Chessboard Squares")
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        return square_images

def loadBinaryModel(model_path,weight_path):
    model = tf.keras.models.load_model(model_path)
    model.load_weights(weight_path)
    return model

def expand_square_if_not_empty(model, warped, sq, margin=2,colour=False,farChessTable=False):
    """
    Uses a binary model to decide if a square is empty. If not, expand the square
    to fully include any piece detected via edge detection.

    Parameters:
        model: TensorFlow binary classifier (empty vs occupied)
        warped: top-down chessboard image (BGR or grayscale)
        sq: 4 corner points of the square (np.array of shape (4,2))
        margin: extra pixels to add around detected piece edges

    Returns:
        square_crop: resized 64x64 image of the (expanded if needed) square
    """
    sq = sq.astype(int)
    xs = sq[:, 0]
    ys = sq[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_min_clipped = max(x_min, 0)
    y_min_clipped = max(y_min, 0)
    x_max_clipped = min(x_max, warped.shape[1])
    y_max_clipped = min(y_max, warped.shape[0])

    square_crop = warped[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped]

    # Binary model check
    square_input = cv2.resize(square_crop, (64, 64))
    square_input = cv2.cvtColor(square_input, cv2.COLOR_BGR2GRAY)
    norm_sq = square_input / 255.0
    norm_sq = norm_sq[..., None]
    norm_sq = np.expand_dims(norm_sq, axis=0)

    pred = model.predict(norm_sq, verbose=0)[0][0]

    if pred < 0.5:
        if colour:
            return cv2.resize(square_crop, (128, 128))
        return cv2.resize(square_crop, (64, 64))

    # --- Occupied: BFS flood fill from centre to find piece bounds ---
    if len(warped.shape) == 3:
        img_gray = warped.copy()
    else:
        img_gray = warped.copy()

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    square_height = y_max - y_min
    square_width = x_max - x_min

    
    patch_size = 5  # sample a 5x5 patch
    half = patch_size // 2
    patch = img_gray[
        max(cy - half, 0):min(cy + half + 1, warped.shape[0]),
        max(cx - half, 0):min(cx + half + 1, warped.shape[1])
    ]
    piece_colour = patch.reshape(-1, 3).mean(axis=0).astype(float)

    deviation_threshold = 18

    # BFS flood fill bounded to 3x square size search area
    search_y_min = max(cy - 0.5 * square_height, 0)
    search_y_max = min(cy + 0.5 * square_height, warped.shape[0])
    search_x_min = max(cx - 0.5 * square_width, 0)
    search_x_max = min(cx + 0.5 * square_width, warped.shape[1])

    visited = np.zeros(warped.shape[:2], dtype=bool)
    queue = [(cy, cx)]
    visited[cy, cx] = True

    piece_top = cy
    piece_bottom = cy
    piece_left = cx
    piece_right = cx

    # 8-connected neighbourhood
    neighbours = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while queue:
        y, x = queue.pop(0)

        # Update bounding box
        piece_top = min(piece_top, y)
        piece_bottom = max(piece_bottom, y)
        piece_left = min(piece_left, x)
        piece_right = max(piece_right, x)

        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx

            # Stay within search bounds
            if not (search_y_min <= ny < search_y_max and search_x_min <= nx < search_x_max):
                continue
            if visited[ny, nx]:
                continue
            pixel = img_gray[ny, nx].astype(float)
            if np.linalg.norm(pixel - piece_colour) <= deviation_threshold:
                # print("Deviating pixel found at", (ny, nx), "with value", np.linalg.norm(pixel - piece_colour))
                visited[ny, nx] = True
                queue.append((ny, nx))
    
    # Only expand if piece extends beyond current square bounds
    new_y_min = piece_top if piece_top < y_min else y_min
    new_y_max = piece_bottom if piece_bottom > y_max else y_max
    new_x_min = piece_left if piece_left < x_min else x_min
    new_x_max = piece_right if piece_right > x_max else x_max

    # vis = warped.copy()
    
    # # Draw all visited pixels in green
    # vis[visited] = [0, 255, 0]
    
    # # Draw rejected pixels in red (within search bounds but outside threshold)
    # for y in range(search_y_min, search_y_max):
    #     for x in range(search_x_min, search_x_max):
    #         if not visited[y, x]:
    #             pixel = img_gray[y, x].astype(float)
    #             if np.linalg.norm(pixel - piece_colour) > deviation_threshold:
    #                 vis[y, x] = [0, 0, 255]

    # # Draw the original square bounds in blue and new bounds in yellow
    # cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)      # blue = original
    # cv2.rectangle(vis, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 255), 2)  # yellow = expanded
    # cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)  # white = centre point

    # plt.figure(figsize=(8, 8))
    # plt.title(f"BFS Fill — threshold={deviation_threshold}")
    # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    
    # Apply margin only where expansion happened
    new_y_min = max(new_y_min - (margin if new_y_min < y_min else 0), 0)
    new_y_max = min(new_y_max + (margin if new_y_max > y_max else 0), warped.shape[0])
    new_x_min = max(new_x_min - (margin if new_x_min < x_min else 0), 0)
    new_x_max = min(new_x_max + (margin if new_x_max > x_max else 0), warped.shape[1])

    square_crop = warped[new_y_min:new_y_max, new_x_min:new_x_max]

    if colour:
        return cv2.resize(square_crop, (128, 128))
    return cv2.resize(square_crop, (64, 64))

# path_opening = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\opening"
# path_midgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\midgame"
# path_endgame = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\endgame"

# path_testing = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\testingImages"

# image_files = [
#     os.path.join(path_testing, f)
#     for f in os.listdir(path_testing)
#     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
# ]


# image_filesO = [
#     os.path.join(path_opening, f)
#     for f in os.listdir(path_opening)
#     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
# ]

# image_filesM = [
#     os.path.join(path_midgame, f)
#     for f in os.listdir(path_midgame)
#     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
# ]

# image_filesE = [
#     os.path.join(path_endgame, f)
#     for f in os.listdir(path_endgame)
#     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
# ]


# image_files = image_filesO + image_filesM + image_filesE

# offset = 0
# for i, image in enumerate(image_files):
#     if i < offset:
#         continue
#     print(f"Processing image {i+1}/{len(image_files)}: {image}")
#     squares = getSquaresFromImage(image, debug=False,colour=True)

# folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\chessRed\\FinalImages"
# image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# image_files = [
#     os.path.join(folder_path, f)
#     for f in os.listdir(folder_path)
#     if f.lower().endswith(image_extensions)
# ]

# images = []
# offset = 35
# for i, image in enumerate(image_files):
#     if i < offset:
#         continue
#     print(f"Processing image {i+1}/{len(image_files)}: {image}")
#     try:
#         squares = getSquaresFromImage(image, debug=False,farChessTable=True)
#     except Exception as e:
#         print(f"Error processing image {image}: {e}")



# folder_path = "C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\trainingData\\fillers"
# image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# image_files = [
#     os.path.join(folder_path, f)
#     for f in os.listdir(folder_path)
#     if f.lower().endswith(image_extensions)
# ]

# images = []
# offset = 34
# for i, image in enumerate(image_files):
#     if i < offset:
#         continue
#     print(f"Processing image {i+1}/{len(image_files)}: {image}")
#     try:
#         squares = getSquaresFromImage(image, debug=False,farChessTable=False)
#     except Exception as e:
#         print(f"Error processing image {image}: {e}")


# getSquaresFromImage("C:\\Users\\Callu\\Documents\\MyDocuments\\University\\Year3\\Dissertation\\Code\\test.jpg",debug=True)


# Method 3: SIFT + DBSCAN + Color Merging

# sift = cv2.SIFT_create()
# keypoints = [cv2.KeyPoint(float(x), float(y), 20) for x, y in filtered_points]

# keypoints, descriptors = sift.compute(grey, keypoints)

# img_kp = img.copy()
# cv2.drawKeypoints(img, keypoints, img_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(6,6))
# plt.title("SIFT Keypoints")
# plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# print("Descriptor shape:", descriptors.shape)
# db = DBSCAN(eps=300,min_samples=2,metric='euclidean')
# labels = db.fit_predict(descriptors)


# points = np.array([kp.pt for kp in keypoints])  # shape (N,2)

# # Assign colors for each cluster
# unique_labels = set(labels)
# colors = plt.cm.jet(np.linspace(0,1,len(unique_labels)))

# img_clusters = img.copy()

# for k, col in zip(unique_labels, colors):
#     class_member_mask = (labels == k)
#     xy = points[class_member_mask]
#     if k == -1:
#         # Noise = black
#         col = [0,0,0,1]
#     for x, y in xy:
#         cv2.circle(img_clusters, (int(x), int(y)), 3, tuple([int(c*255) for c in col[:3]]), -1)

# plt.figure(figsize=(6,6))
# plt.imshow(cv2.cvtColor(img_clusters, cv2.COLOR_BGR2RGB))
# plt.title("DBSCAN Clusters Before Color Merging")
# plt.axis('off')
# plt.show()


# unique_labels = set(labels)
# clusters = {}
# cluster_colours = {}

# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# for k in unique_labels:
#     if k == -1:
#         continue  # Skip noise
#     cluster_kp = [kp for kp, label in zip(keypoints, labels) if label == k]
#     clusters[k] = cluster_kp

#     # Compute average color of cluster
#     pixels = np.array([img_lab[int(kp.pt[1]), int(kp.pt[0])] for kp in cluster_kp])
#     avg_color = np.mean(pixels, axis=0)
#     cluster_colours[k] = avg_color

# merge_done = True
# threshold_color = 40  # you can tune this
# threshold_distance = 250 

# while merge_done:
#     merge_done = False
#     merged_clusters = {}
#     merged_labels = set()

#     for k1 in clusters.keys():
#         if k1 in merged_labels:
#             continue
#         merged = clusters[k1].copy()
#         merged_labels.add(k1)
#         color1 = cluster_colours[k1]

#         # Compute centroid of cluster k1
#         pts1 = np.array([kp.pt for kp in clusters[k1]])
#         centroid1 = pts1.mean(axis=0)

#         for k2 in clusters.keys():
#             if k2 in merged_labels:
#                 continue
#             color2 = cluster_colours[k2]

#             # Compute centroid of cluster k2
#             pts2 = np.array([kp.pt for kp in clusters[k2]])
#             centroid2 = pts2.mean(axis=0)

#             # Compute Euclidean distance between centroids
#             dist = np.linalg.norm(centroid1 - centroid2)

#             # threshold_distance = max(200, (1000 / len(clusters)))
#             if np.linalg.norm(color1 - color2) < threshold_color and dist < threshold_distance:
#                 merged.extend(clusters[k2])
#                 merged_labels.add(k2)
#                 merge_done = True
#         merged_clusters[k1] = merged

#     # Add clusters that were not merged
#     # for k in clusters.keys():
#     #     if k not in merged_labels:
#     #         merged_clusters[k] = clusters[k]
#     #         merged_labels.add(k)

#     # Update clusters for next iteration
#     clusters = merged_clusters.copy() 
#     unique_labels = merged_labels.copy()


#     cluster_colours = {}
#     for k, cluster_kp in clusters.items():
#         # Compute new average color from merged points
#         pixels = np.array([img_lab[int(kp.pt[1]), int(kp.pt[0])] for kp in cluster_kp])
#         cluster_colours[k] = np.mean(pixels, axis=0)

    
#     img_merged = img.copy()

#     # Assign a unique color to each merged cluster
#     merged_colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))


#     for col, (k, cluster_kp) in zip(merged_colors, clusters.items()):
#         # Draw all keypoints
#         for kp in cluster_kp:
#             x, y = kp.pt
#             cv2.circle(img_merged, (int(x), int(y)), 3, tuple([int(c*255) for c in col[:3]]), -1)

#         # Compute centroid of this cluster
#         pts = np.array([kp.pt for kp in cluster_kp])
#         centroid = pts.mean(axis=0)
#         cx, cy = int(centroid[0]), int(centroid[1])

#         # Draw centroid (larger circle)
#         cv2.circle(img_merged, (cx, cy), 6, (255, 255, 255), 2)  # white circle
#         # Optional: label the cluster number
#         cv2.putText(img_merged, str(k), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (255,255,255), 1, cv2.LINE_AA)

#     plt.figure(figsize=(6,6))
#     plt.imshow(cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))
#     plt.title("Merged Clusters with Centroids")
#     plt.axis('off')
#     plt.show()



# img_merged = img.copy()

# # Assign a unique color to each merged cluster
# merged_colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))


# for col, (k, cluster_kp) in zip(merged_colors, clusters.items()):
#     # Draw all keypoints
#     for kp in cluster_kp:
#         x, y = kp.pt
#         cv2.circle(img_merged, (int(x), int(y)), 3, tuple([int(c*255) for c in col[:3]]), -1)

#     # Compute centroid of this cluster
#     pts = np.array([kp.pt for kp in cluster_kp])
#     centroid = pts.mean(axis=0)
#     cx, cy = int(centroid[0]), int(centroid[1])

#     # Draw centroid (larger circle)
#     cv2.circle(img_merged, (cx, cy), 6, (255, 255, 255), 2)  # white circle
#     # Optional: label the cluster number
#     cv2.putText(img_merged, str(k), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.5, (255,255,255), 1, cv2.LINE_AA)

# plt.figure(figsize=(6,6))
# plt.imshow(cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))
# plt.title("Merged Clusters with Centroids")
# plt.axis('off')
# plt.show()



# largest_cluster = max(clusters.values(), key=lambda x: len(x))
# board_points = np.array([kp.pt for kp in largest_cluster], dtype=np.int32)

# # --- Convex hull and board extraction ---
# hull = cv2.convexHull(board_points)
# img_hull = img.copy()
# cv2.polylines(img_hull, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

# plt.figure(figsize=(6,6))
# plt.title("Convex Hull of Board Keypoints (Color Merged)")
# plt.imshow(cv2.cvtColor(img_hull, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.fillConvexPoly(mask, hull, 255)

# result = cv2.bitwise_and(img, img, mask=mask)

# plt.figure(figsize=(6,6))
# plt.title("Final Detected Board")
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()




#Method 1!!!!

# sift = cv2.SIFT_create()
# keypoints = [cv2.KeyPoint(float(x), float(y), 20) for x, y in corners]

# keypoints, descriptors = sift.compute(grey, keypoints)

# img_kp = img.copy()
# cv2.drawKeypoints(img, keypoints, img_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.figure(figsize=(6,6))
# plt.title("SIFT Keypoints")
# plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# print("Descriptor shape:", descriptors.shape)
# db = DBSCAN(eps=300,min_samples=2,metric='euclidean')
# labels = db.fit_predict(descriptors)

# plt.figure(figsize=(6,6))
# unique_labels = set(labels)
# colors = plt.cm.jet(np.linspace(0,1,len(unique_labels)))

# for k, col in zip(unique_labels, colors):
#     class_member_mask = (labels == k)
#     xy = descriptors[class_member_mask]
#     if k == -1:
#         # Noise = black
#         col = [0,0,0,1]
#     plt.scatter(xy[:,0], xy[:,1], c=[col], s=20, label=f"Cluster {k}")

# plt.gca().invert_yaxis()  # invert y to match image coordinates
# plt.title("DBSCAN clusters of keypoints")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.show()

# unique_labels, counts = np.unique(labels, return_counts=True)

# cluster_labels = unique_labels[unique_labels != -1]  # Exclude noise label (-1)
# if len(cluster_labels) == 0:
#     print("No clusters found.")
#     board_keypoints = []

# else:

#     largest_cluster_label = unique_labels[np.argmax(counts[unique_labels != -1])]

#     board_keypoints = [kp for kp, label in zip(keypoints, labels) if label == largest_cluster_label]

# print(f"Number of keypoints in largest cluster: {len(board_keypoints)}")

# board_points = np.array([kp.pt for kp in board_keypoints], dtype=np.int32)


# hull = cv2.convexHull(board_points)

# img_hull = img.copy()
# cv2.polylines(img_hull, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

# plt.figure(figsize=(6,6))
# plt.title("Convex Hull of Board Keypoints")
# plt.imshow(cv2.cvtColor(img_hull, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.fillConvexPoly(mask, hull, 255)

# result = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow("Detected Board", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




#Method 2!!!!!
# # Image center
# h, w = grey.shape
# center = np.array([w/2, h/2])

# # Compute distance of each keypoint from center
# distances = np.linalg.norm(filtered_corners - center, axis=1)

# # Keep, for example, the closest 80% of points
# num_points = int(len(filtered_corners) * 1)
# indices = np.argsort(distances)[:num_points]
# board_points = filtered_corners[indices]


# board_hull = cv2.convexHull(board_points)

# # Draw on image
# img_hull = img.copy()
# cv2.polylines(img_hull, [board_hull], isClosed=True, color=(0,255,0), thickness=2)

# plt.figure(figsize=(6,6))
# plt.title("Convex Hull around central keypoints")
# plt.imshow(cv2.cvtColor(img_hull, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.fillConvexPoly(mask, board_hull, 255)
# board_only = cv2.bitwise_and(img, img, mask=mask)

# plt.figure(figsize=(6,6))
# plt.title("Extracted Chessboard")
# plt.imshow(cv2.cvtColor(board_only, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

import math
import os
import cv2
import numpy as np
import itertools
from matplotlib import pyplot as plt
import tensorflow as tf
import BinaryClassificationModel

def getCorners(img, blur, debug):
        blur_float = np.float32(blur)
        initialCorners = cv2.cornerHarris(blur_float,4,3,0.05)
        initialCorners = cv2.dilate(initialCorners, None)

        thresh = 0.025 * initialCorners.max()
        corners = np.argwhere(initialCorners > thresh)

        corners = np.flip(corners, axis=1)  # Flip to (x, y) format

        img_corners = img.copy()
        for x, y in corners:
            cv2.circle(img_corners, (x, y), 3, (0, 0, 255), -1)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Detected Corners")
            plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        responses = initialCorners[corners[:,1], corners[:,0]]
        return responses, corners

def filterPointsBasedOnDistance(points,distance):
        min_dist = distance
        
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

        k = 1.5
        mask = distances < (mean_dist + k * std_dist)

        filtered_points = points[mask]
        return filtered_points, cx, cy

def extractRegionOfInterest(img, corners):
        hull = cv2.convexHull(corners.astype(np.int32))

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        roi = cv2.bitwise_and(img, img, mask=mask)
        return roi

def getHorizontalAndVerticalLines(lines,angle_thresh):
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)

            if abs(angle) < angle_thresh: # near horizontal
                h_lines.append((x1, y1, x2, y2))
            elif abs(abs(angle) - np.pi/2) < angle_thresh:  # near vertical
                v_lines.append((x1, y1, x2, y2))
        return h_lines, v_lines

def intersect(line1, line2):

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        C1 = (y2-y1) * x1 + (x1-x2) * y1
        C2 = (y4-y3) * x3 + (x3-x4) * y3

        det = (y2-y1) * (x3-x4) - (x1-x2) * (y4-y3)
        if abs(det) < 1.01e-6:
            return None  # parallel lines

        x = ((x3-x4) * C1 - (x1-x2) * C2) / det
        y = ((y2-y1) * C2 - (y4-y3) * C1) / det
        return int(x), int(y)

def extendAndClipLines(h_lines, v_lines, warped, debug):
        def extend_line(line, img_shape):
            x1, y1, x2, y2 = line
            h, w = img_shape[:2]

            if x1 == x2:  # vertical
                return (x1, 0, x1, h-1)

            if y1 == y2:  # horizontal
                return (0, y1, w-1, y1)

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            points = []

            # Left border (x = 0)
            y = c
            if 0 <= y < h:
                points.append((0, int(y)))

            # Right border (x = w-1)
            y = m * (w-1) + c
            if 0 <= y < h:
                points.append((w-1, int(y)))

            # Top border (y = 0)
            x = -c / m
            if 0 <= x < w:
                points.append((int(x), 0))

            # Bottom border (y = h-1)
            x = (h-1 - c) / m
            if 0 <= x < w:
                points.append((int(x), h-1))

            if len(points) >= 2:
                return (*points[0], *points[1])

            return line

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

            x1, y1, x2, y2 = line

            if y2 == y1:
                return line

            m = (x2 - x1) / (y2 - y1)

            x_top = x1 + (y_top - y1) * m
            x_bottom = x1 + (y_bottom - y1) * m

            return (int(x_top), int(y_top), int(x_bottom), int(y_bottom))

        def clip_horizontal_line(line, x_left, x_right):
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

        v_lines_clipped = [clip_vertical_line(l, y_top, y_bottom) for l in v_lines_ext]
        h_lines_clipped = [clip_horizontal_line(l, x_left, x_right) for l in h_lines_ext]
        return h_lines_clipped, v_lines_clipped

def selectNineBestLines(lines, axis, img_size, warped, angle_tol_deg=20):
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

            #Get rid of outliers based on angle consistency to improve performance
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
def getBoardFromImage(img, debug=False, farChessTable=False):
        h, w = img.shape[:2]

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        
        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Preprocessed Grayscale Image")
            plt.imshow(grey, cmap='gray')
            plt.axis('off')
            plt.show()

        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Blurred Image")
            plt.imshow(blur, cmap='gray')
            plt.axis('off')
            plt.show()
        
        responses,corners = getCorners(img, blur, debug)
        idx = np.argsort(responses)[::-1]  # reorder corners by strength
        points = corners[idx]


        filtered_points, cx, cy = filterPointsBasedOnDistance(points,10)

        if debug:
            #show filtered points
            img_filtered = img.copy()
            for x, y in filtered_points:
                cv2.circle(img_filtered, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(img_filtered, (cx, cy), 8, (0, 255, 255), -1)
            cv2.putText(img_filtered, "Mean", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            plt.figure(figsize=(6,6))
            plt.title("Filtered Corners")
            plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        if farChessTable:
            grey_roi = cv2.cvtColor(extractRegionOfInterest(img, filtered_points), cv2.COLOR_BGR2GRAY)
        else:
            grey_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey_roi, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=40,
                maxLineGap=20
        )

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

        h_lines, v_lines = getHorizontalAndVerticalLines(lines,np.pi / 12)

        top = min(h_lines, key=lambda l: min(l[1], l[3]))
        bottom = max(h_lines, key=lambda l: max(l[1], l[3]))
        left = min(v_lines, key=lambda l: min(l[0], l[2]))
        right = max(v_lines, key=lambda l: max(l[0], l[2]))

        if debug:
            img_lines = img.copy()
            for l in [top, bottom, left, right]:
                cv2.line(img_lines, (l[0],l[1]), (l[2],l[3]), (0,255,0), 3)

            plt.figure(figsize=(6,6))
            plt.title("Selected Board Lines")
            plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

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
        return warped

def getSquaresFromImage(image_path,debug=False,farChessTable=False,noModifySquares=False):
        model = None
        if not noModifySquares:
            model = loadBinaryModel("binary_classification_model.h5", "binary_classification_weights.h5")
            if(model is None):
                print("Binary classification model not found. Training a new model...")
                BinaryClassificationModel.trainAndSaveModel()
                model = loadBinaryModel("binary_classification_model.h5", "binary_classification_weights.h5")
                if(model is None):
                    print("Failed to load or train binary classification model.")
                    return None, None
        img = cv2.imread(image_path)
        warped = getBoardFromImage(img, debug=debug, farChessTable=farChessTable)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Warped Chessboard (Top-Down View)")
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        grey_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        grey_warped = cv2.GaussianBlur(grey_warped, (5, 5), 0)

        edges = cv2.Canny(grey_warped, 50, 150)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Edges on Warped Board")
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Merged Edges on Warped Board")
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


        h_lines, v_lines = getHorizontalAndVerticalLines(lines,np.pi / 10)

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

        def merge_similar_lines(lines, axis, dist_thresh):
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
                    len_new = np.hypot(line[2]-line[0], line[3]-line[1])
                    len_old = np.hypot(prev[2]-prev[0], prev[3]-prev[1])

                    if len_new > len_old:
                        merged[-1] = line
                else:
                    merged.append(line)

            return merged

        h_lines = merge_similar_lines(h_lines, 'y', 15)
        v_lines = merge_similar_lines(v_lines, 'x', 15)

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


        h_lines_clipped, v_lines_clipped = extendAndClipLines(h_lines, v_lines, warped, debug=debug)

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


        v_grid = selectNineBestLines(v_lines, 'x', warped.shape[1], warped)
        h_grid = selectNineBestLines(h_lines, 'y', warped.shape[0], warped)

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

        for i in range(8):
            for j in range(8):
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

            if noModifySquares:
                square_images.append(cv2.resize(square_img, (128, 128)))
                continue
            square_img =  expand_square_if_not_empty(model,warped, sq, margin=50)

            square_images.append(square_img)

        if debug:
            plt.figure(figsize=(6,6))
            plt.title("Detected Chessboard Squares")
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        return square_images

def loadBinaryModel(model_path,weight_path):
    try:
        model = tf.keras.models.load_model(model_path)
        model.load_weights(weight_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def expand_square_if_not_empty(model, warped, sq, margin=2):

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

    square_input = cv2.resize(square_crop, (64, 64))
    square_input = cv2.cvtColor(square_input, cv2.COLOR_BGR2GRAY)
    norm_sq = square_input / 255.0
    norm_sq = norm_sq[..., None]
    norm_sq = np.expand_dims(norm_sq, axis=0)

    pred = model.predict(norm_sq, verbose=0)[0][0]

    if pred < 0.5:
        return cv2.resize(square_crop, (128, 128))


    if len(warped.shape) == 3:
        img_gray = warped.copy()
    else:
        img_gray = warped.copy()

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    square_height = y_max - y_min
    square_width = x_max - x_min

    
    patch_size = 5 
    half = patch_size // 2
    patch = img_gray[
        max(cy - half, 0):min(cy + half + 1, warped.shape[0]),
        max(cx - half, 0):min(cx + half + 1, warped.shape[1])
    ]
    piece_colour = patch.reshape(-1, 3).mean(axis=0).astype(float)

    deviation_threshold = 18

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
                visited[ny, nx] = True
                queue.append((ny, nx))
    
    new_y_min = y_min
    new_y_max = y_max
    new_x_min = x_min
    new_x_max = x_max

    # Expand bounds based on piece position
    if piece_top < y_min:
        new_y_min = piece_top
    if piece_bottom > y_max:
        new_y_max = piece_bottom
    if piece_left < x_min:
        new_x_min = piece_left
    if piece_right > x_max:
        new_x_max = piece_right

    
    new_y_min = max(new_y_min - (margin if new_y_min < y_min else 0), 0)
    new_y_max = min(new_y_max + (margin if new_y_max > y_max else 0), warped.shape[0])
    new_x_min = max(new_x_min - (margin if new_x_min < x_min else 0), 0)
    new_x_max = min(new_x_max + (margin if new_x_max > x_max else 0), warped.shape[1])

    square_crop = warped[new_y_min:new_y_max, new_x_min:new_x_max]


    return cv2.resize(square_crop, (128, 128))


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
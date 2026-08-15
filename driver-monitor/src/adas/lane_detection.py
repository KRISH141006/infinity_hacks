# lane_detection.py - Advanced Lane Line Detection & Departure Warning
import cv2
import numpy as np
try:
    from .config import PROCESS_WIDTH, PROCESS_HEIGHT, LANE_ROI, LANE_DEVIATION_THRESHOLD
except ImportError:
    from config import PROCESS_WIDTH, PROCESS_HEIGHT, LANE_ROI, LANE_DEVIATION_THRESHOLD

class AdvancedLaneDetector:
    def __init__(self, width=PROCESS_WIDTH, height=PROCESS_HEIGHT):
        self.width = width
        self.height = height
        
        # Define source points for perspective transform (trapezoid on road)
        self.src_pts = np.float32([
            [int(LANE_ROI[0][0] * self.width), int(LANE_ROI[0][1] * self.height)], # Bottom-Left
            [int(LANE_ROI[1][0] * self.width), int(LANE_ROI[1][1] * self.height)], # Top-Left
            [int(LANE_ROI[2][0] * self.width), int(LANE_ROI[2][1] * self.height)], # Top-Right
            [int(LANE_ROI[3][0] * self.width), int(LANE_ROI[3][1] * self.height)]  # Bottom-Right
        ])
        
        # Define destination points (rectangle top-down representation)
        self.dst_pts = np.float32([
            [int(self.width * 0.2), self.height],
            [int(self.width * 0.2), 0],
            [int(self.width * 0.8), 0],
            [int(self.width * 0.8), self.height]
        ])
        
        # Compute perspective matrices
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        
        # Historical coefficients for moving average smoothing (last 10 frames)
        self.left_fit_history = []
        self.right_fit_history = []
        self.max_history = 10
        
        self.last_left_fit = None
        self.last_right_fit = None

    def threshold_frame(self, frame):
        """Applies HLS color filtering and Sobel gradient to isolate yellow/white lanes."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        # Saturation channel threshold (good for yellow lines)
        s_binary = cv2.threshold(s_channel, 120, 255, cv2.THRESH_BINARY)[1]
        
        # Lightness channel threshold (good for white lines)
        l_binary = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)[1]
        
        # Sobel X-gradient on Lightness (detects vertical transitions)
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)
        max_sobel = np.max(abs_sobel_x)
        if max_sobel > 0:
            scaled_sobel = np.uint8(255 * abs_sobel_x / max_sobel)
        else:
            scaled_sobel = np.zeros_like(l_channel)
        sobel_binary = cv2.threshold(scaled_sobel, 35, 255, cv2.THRESH_BINARY)[1]
        
        # Combine binary thresholds
        combined_binary = np.zeros_like(s_binary)
        combined_binary[((s_binary == 255) | (l_binary == 255) | (sobel_binary == 255))] = 255
        
        return combined_binary

    def warp(self, frame):
        """Transforms perspective to bird's-eye view."""
        return cv2.warpPerspective(frame, self.M, (self.width, self.height))

    def fit_polynomial(self, binary_warped):
        """Uses sliding windows to trace lanes and fits second-order polynomials."""
        # Find peaks in bottom half histogram to find starting points
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        
        # Left and right starting points
        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint
        
        # Sliding windows parameters
        nwindows = 9
        window_height = int(binary_warped.shape[0] // nwindows)
        margin = 80
        minpix = 40
        
        # Get active pixel coordinates
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = left_peak
        rightx_current = right_peak
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            return self.last_left_fit, self.last_right_fit
            
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if len(leftx) > 80 and len(lefty) > 80:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_fit_history.append(left_fit)
            if len(self.left_fit_history) > self.max_history:
                self.left_fit_history.pop(0)
            left_fit = np.mean(self.left_fit_history, axis=0)
            self.last_left_fit = left_fit
        else:
            left_fit = self.last_left_fit
            
        if len(rightx) > 80 and len(righty) > 80:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_fit_history.append(right_fit)
            if len(self.right_fit_history) > self.max_history:
                self.right_fit_history.pop(0)
            right_fit = np.mean(self.right_fit_history, axis=0)
            self.last_right_fit = right_fit
        else:
            right_fit = self.last_right_fit
            
        return left_fit, right_fit

    def project_lanes(self, frame, left_fit, right_fit):
        """Draws the lane area and overlays back onto the original perspective."""
        if left_fit is None or right_fit is None:
            return frame, "Lanes Unclear", (128, 128, 128), 0.0
            
        ploty = np.linspace(0, self.height - 1, self.height)
        
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        warp_zero = np.zeros((self.height, self.width), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw green drivable path
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 0))
        
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (self.width, self.height))
        result = cv2.addWeighted(frame, 1.0, newwarp, 0.25, 0)
        
        # Left boundary line
        left_pts = np.int32(np.transpose(np.vstack([left_fitx, ploty])))
        left_pts_orig = cv2.perspectiveTransform(left_pts.reshape(-1, 1, 2).astype(np.float32), self.M_inv)
        cv2.polylines(result, [np.int32(left_pts_orig)], isClosed=False, color=(0, 255, 0), thickness=3)
        
        # Right boundary line
        right_pts = np.int32(np.transpose(np.vstack([right_fitx, ploty])))
        right_pts_orig = cv2.perspectiveTransform(right_pts.reshape(-1, 1, 2).astype(np.float32), self.M_inv)
        cv2.polylines(result, [np.int32(right_pts_orig)], isClosed=False, color=(0, 255, 0), thickness=3)
        
        # Lane departure deviation
        car_position = self.width / 2.0
        lane_left_bottom = left_fit[0] * self.height**2 + left_fit[1] * self.height + left_fit[2]
        lane_right_bottom = right_fit[0] * self.height**2 + right_fit[1] * self.height + right_fit[2]
        lane_center = (lane_left_bottom + lane_right_bottom) / 2.0
        
        deviation_px = lane_center - car_position
        
        if deviation_px > LANE_DEVIATION_THRESHOLD:
            msg = "WARNING: Lane Drift Right!"
            color = (0, 165, 255)
        elif deviation_px < -LANE_DEVIATION_THRESHOLD:
            msg = "WARNING: Lane Drift Left!"
            color = (0, 165, 255)
        else:
            msg = "Lane Center"
            color = (0, 255, 0)
            
        return result, msg, color, float(deviation_px)

    def process(self, frame):
        """Top-level process wrapper."""
        warped = self.warp(frame)
        binary_warped = self.threshold_frame(warped)
        left_fit, right_fit = self.fit_polynomial(binary_warped)
        output_frame, msg, color, deviation_px = self.project_lanes(frame, left_fit, right_fit)
        return output_frame, msg, color, deviation_px

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from scipy.signal import savgol_filter

class EngagementAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.FACE_3D_MODEL = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
        ], dtype=np.float64)
        self.HEAD_IDXS = [1, 152, 33, 263, 61, 291]
        self.LEFT_EYE = [33, 133, 160, 158, 144, 153]
        self.RIGHT_EYE = [362, 263, 385, 387, 380, 373]
        self.MOUTH = [78, 308, 13, 14]
        self.RIGHT_IRIS = [33, 133, 468]
        self.LEFT_IRIS = [362, 263, 473]

    def _get_head_pose(self, landmarks, w, h):
        face_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.HEAD_IDXS], dtype=np.float64)
        eye_dist = np.linalg.norm(face_2d[2] - face_2d[3])
        scale = eye_dist / 86.6
        face_3d_scaled = self.FACE_3D_MODEL * scale
        focal = w
        cam_matrix = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1))
        success, rvec, _ = cv2.solvePnP(face_3d_scaled, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0], angles[1], angles[2]

    def _get_gaze_ratio(self, landmarks, indices, w, h):
        p_inner = np.array([landmarks[indices[0]].x * w, landmarks[indices[0]].y * h])
        p_outer = np.array([landmarks[indices[1]].x * w, landmarks[indices[1]].y * h])
        p_iris  = np.array([landmarks[indices[2]].x * w, landmarks[indices[2]].y * h])
        eye_width = np.linalg.norm(p_outer - p_inner)
        if eye_width == 0: return 0.5, 0 
        dist_to_inner = np.linalg.norm(p_iris - p_inner)
        return dist_to_inner / eye_width, 1

    def _calculate_ear(self, landmarks, indices, w, h):
        coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        d_ver = (np.linalg.norm(coords[2]-coords[4]) + np.linalg.norm(coords[3]-coords[5])) / 2
        d_hor = np.linalg.norm(coords[0]-coords[1])
        return d_ver / (d_hor + 1e-6)
    
    def _calculate_mar(self, landmarks, indices, w, h):
        coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        d_ver = np.linalg.norm(coords[2] - coords[3])
        d_hor = np.linalg.norm(coords[0] - coords[1])
        return d_ver / (d_hor + 1e-6)

    def _advanced_smooth(self, data, window_size):
        data = np.array(data)
        window_length = int(window_size)
        if window_length % 2 == 0: window_length += 1
        if len(data) < window_length: return data
        return savgol_filter(data, window_length, polyorder=2)

    def _get_robust_base(self, data, bins=30):
        data = np.array(data)
        data = data[~np.isnan(data)]
        if len(data) == 0: return 0
        hist, bin_edges = np.histogram(data, bins=bins)
        max_idx = np.argmax(hist)
        return (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2

    def _classify_session(self, score, breakdown):
        PASS_THRESHOLD = 50.0 
        
        if score >= PASS_THRESHOLD:
            label = "ON_TASK"
            reason = "Good engagement level"
        else:
            label = "OFF_TASK"
            max_cause = max(breakdown, key=breakdown.get)
            if breakdown[max_cause] > 0:
                reason = f"Mainly due to {max_cause} ({breakdown[max_cause]:.1f}s)"
            else:
                reason = "General distraction"
                
        return label, reason

    def process_video(self, video_path, output_folder):
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0

        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        process_w = 640
        scale_factor = process_w / original_w
        process_h = int(original_h * scale_factor)
        
        print(f"Analyzing: {video_path} | Duration: {video_duration:.1f}s")

        raw_yaw, raw_pitch, raw_ear, raw_mar, raw_gaze = [], [], [], [], []
        timestamps, frame_indices = [], []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_resized = cv2.resize(frame, (process_w, process_h))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            y, p, e, m, g = np.nan, np.nan, np.nan, np.nan, np.nan
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                pose = self._get_head_pose(lms, process_w, process_h)
                if pose: p, y, _ = pose
                e = (self._calculate_ear(lms, self.LEFT_EYE, process_w, process_h) + 
                     self._calculate_ear(lms, self.RIGHT_EYE, process_w, process_h)) / 2.0
                m = self._calculate_mar(lms, self.MOUTH, process_w, process_h)
                gr, vr = self._get_gaze_ratio(lms, self.RIGHT_IRIS, process_w, process_h)
                gl, vl = self._get_gaze_ratio(lms, self.LEFT_IRIS, process_w, process_h)
                if vr and vl: g = (gr + gl) / 2.0

            raw_yaw.append(y); raw_pitch.append(p)
            raw_ear.append(e); raw_mar.append(m); raw_gaze.append(g)
            timestamps.append(frame_idx / fps if fps else 0)
            frame_indices.append(frame_idx)
            frame_idx += 1
        cap.release()

        def fill_nan(arr):
            arr = np.array(arr)
            mask = np.isnan(arr)
            if np.sum(~mask) < 2: return arr 
            arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
            return arr

        if np.all(np.isnan(raw_yaw)): return {"status": "FAILED", "reason": "No face detected"}

        win_short = max(3, int(fps * 0.3)) 
        win_long = max(5, int(fps * 1.0))

        yaw_s = self._advanced_smooth(fill_nan(raw_yaw), win_long)
        pitch_s = self._advanced_smooth(fill_nan(raw_pitch), win_long)
        ear_s = self._advanced_smooth(fill_nan(raw_ear), win_short) 
        mar_s = self._advanced_smooth(fill_nan(raw_mar), win_short)
        gaze_s = self._advanced_smooth(fill_nan(raw_gaze), win_long)

        sorted_ear = np.sort(ear_s)
        top_20_idx = int(len(sorted_ear) * 0.8)
        EAR_BASE = np.median(sorted_ear[top_20_idx:]) if top_20_idx < len(sorted_ear) else 0.3

        temp_yaw_median = np.nanmedian(yaw_s)
        temp_pitch_median = np.nanmedian(pitch_s)
        head_stable_indices = (np.abs(yaw_s - temp_yaw_median) < 15) & \
                              (np.abs(pitch_s - temp_pitch_median) < 15)
        valid_gaze = gaze_s[head_stable_indices]
        GAZE_BASE = self._get_robust_base(valid_gaze) if len(valid_gaze) > 10 else self._get_robust_base(gaze_s)
        
        YAW_BASE = self._get_robust_base(yaw_s)
        PITCH_BASE = self._get_robust_base(pitch_s)

        H_ENTER = max(20.0, np.std(yaw_s) * 2.5)
        H_EXIT  = H_ENTER * 0.75 
        G_ENTER = max(0.12, np.std(gaze_s) * 2.2)
        G_EXIT  = G_ENTER * 0.8
        THRESH_EAR = EAR_BASE * 0.70

        timeline = []
        evidence_frames = []
        current_event = None
        
        sleep_cnt, noface_cnt = 0, 0
        flag_head, flag_gaze = False, False
        LIMIT_SLEEP = int(fps * 1.5)
        LIMIT_NOFACE = int(fps * 1.0)

        for i in range(len(timestamps)):
            if np.isnan(raw_yaw[i]): noface_cnt += 1
            else: noface_cnt = 0
            is_noface = noface_cnt > LIMIT_NOFACE

            if is_noface:
                status = "NO_FACE"
                flag_head = False; flag_gaze = False; sleep_cnt = 0
            else:
                if ear_s[i] < THRESH_EAR: sleep_cnt += 1
                else: sleep_cnt = 0
                is_sleep = sleep_cnt > LIMIT_SLEEP

                dev_head = max(abs(yaw_s[i] - YAW_BASE), abs(pitch_s[i] - PITCH_BASE))
                if not flag_head:
                    if dev_head > H_ENTER: flag_head = True
                else:
                    if dev_head < H_EXIT: flag_head = False
                
                dev_gaze = abs(gaze_s[i] - GAZE_BASE)
                if not flag_head and not is_sleep and not np.isnan(raw_gaze[i]):
                    if not flag_gaze:
                        if dev_gaze > G_ENTER: flag_gaze = True
                    else:
                        if dev_gaze < G_EXIT: flag_gaze = False
                else:
                    flag_gaze = False

                if is_sleep: status = "SLEEPING"
                elif flag_head: status = "HEAD_AWAY"
                elif flag_gaze: status = "DISTRACTED"
                else: status = "ON_TASK"

            is_off_task = status in ["SLEEPING", "HEAD_AWAY", "DISTRACTED", "NO_FACE"]
            if is_off_task:
                if current_event is None:
                    current_event = {'start_time': timestamps[i], 'start_frame': frame_indices[i], 'type': status}
                elif current_event['type'] != status:
                    current_event['end_time'] = timestamps[i-1]
                    current_event['end_frame'] = frame_indices[i-1]
                    if (current_event['end_time'] - current_event['start_time']) > 1.0:
                        timeline.append(current_event)
                        mid = int((current_event['start_frame'] + current_event['end_frame']) / 2)
                        evidence_frames.append((mid, current_event['type']))
                    current_event = {'start_time': timestamps[i], 'start_frame': frame_indices[i], 'type': status}
            else:
                if current_event:
                    current_event['end_time'] = timestamps[i-1]
                    current_event['end_frame'] = frame_indices[i-1]
                    if (current_event['end_time'] - current_event['start_time']) > 1.0:
                        timeline.append(current_event)
                        mid = int((current_event['start_frame'] + current_event['end_frame']) / 2)
                        evidence_frames.append((mid, current_event['type']))
                    current_event = None

        if current_event:
            current_event['end_time'] = timestamps[-1]
            current_event['end_frame'] = frame_indices[-1]
            if (current_event['end_time'] - current_event['start_time']) > 1.0:
                timeline.append(current_event)
                mid = int((current_event['start_frame'] + current_event['end_frame']) / 2)
                evidence_frames.append((mid, current_event['type']))

        cap = cv2.VideoCapture(video_path)
        saved_images = []
        evidence_frames.sort(key=lambda x: x[0])
        for f_idx, label in evidence_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                fname = f"ev_{f_idx}_{label}.jpg"
                fpath = os.path.join(output_folder, fname)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0,0,0), -1)
                cv2.putText(frame, f"EVIDENCE: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imwrite(fpath, frame)
                saved_images.append({'time': f_idx/fps if fps else 0, 'label': label, 'path': fpath})
        cap.release()

        total_penalty = 0
        breakdown = {"SLEEPING": 0, "HEAD_AWAY": 0, "DISTRACTED": 0, "NO_FACE": 0}
        
        for e in timeline:
            dur = e['end_time'] - e['start_time']
            if e['type'] in breakdown:
                breakdown[e['type']] += dur
            
            if e['type'] == "SLEEPING": mult = 1.5
            elif e['type'] == "HEAD_AWAY": mult = 1.0
            elif e['type'] == "NO_FACE": mult = 1.5
            elif e['type'] == "DISTRACTED": mult = 0.6
            else: mult = 0.5
            
            total_penalty += dur * mult

        if video_duration > 0:
            penalty_ratio = total_penalty / video_duration
            final_score = 100 * np.exp(-2.0 * penalty_ratio)
            final_score = max(0, min(100, final_score))
        else:
            final_score = 0

        final_label, main_reason = self._classify_session(final_score, breakdown)

        return {
            "final_label": final_label,
            "score": round(final_score, 1),
            "reason": main_reason,
            "duration": round(video_duration, 1),
            "breakdown": {k: round(v, 1) for k, v in breakdown.items()},
            "timeline": timeline,
            "evidence": saved_images
        }
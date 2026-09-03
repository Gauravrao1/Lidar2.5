from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

@dataclass
class TrackedObject:
    """A tracked dynamic object across frames."""
    track_id: int
    centroid: np.ndarray  # (3,) current position
    velocity: np.ndarray  # (3,) estimated velocity (m/frame)
    speed_mps: float  # scalar speed in m/s (assuming 10 FPS)
    bbox_min: np.ndarray  # (3,) bounding box minimum corner
    bbox_max: np.ndarray  # (3,) bounding box maximum corner
    num_points: int
    first_seen_frame: int
    last_seen_frame: int
    age: int  # frames since first seen
    hits: int  # number of frames where this object was detected
    history: List[np.ndarray]  # list of past centroids for trajectory

class DynamicObjectTracker:
    """Tracks dynamic objects across LiDAR frames using nearest-centroid association.
    
    Algorithm:
    1. Cluster dynamic points (class=2) using simple distance-based grouping
    2. Compute centroid of each cluster
    3. Match to existing tracks using Hungarian/greedy nearest-neighbor
    4. Update matched tracks with new position, compute velocity
    5. Create new tracks for unmatched detections
    6. Mark tracks as lost if not seen for max_lost_frames
    """
    
    def __init__(self, max_association_dist: float = 5.0, max_lost_frames: int = 5,
                 min_cluster_points: int = 10, cluster_eps: float = 1.5,
                 assumed_fps: float = 10.0):
        self.max_assoc_dist = max_association_dist
        self.max_lost = max_lost_frames
        self.min_cluster_pts = min_cluster_points
        self.cluster_eps = cluster_eps
        self.assumed_fps = assumed_fps
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0
        self._total_tracks_ever = 0
    
    def cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """Simple grid-based clustering of 3D points.
        Divide space into cluster_eps-sized voxels, group connected voxels.
        Return list of point arrays, one per cluster.
        Only return clusters with >= min_cluster_points."""
        if len(points) == 0:
            return []
            
        # Quantize points to voxels
        voxel_indices = np.floor(points / self.cluster_eps).astype(np.int32)
        
        # Create a mapping from voxel index tuple to point indices
        voxel_map: Dict[Tuple[int, int, int], List[int]] = {}
        for i, vi in enumerate(voxel_indices):
            v_tuple = (vi[0], vi[1], vi[2])
            if v_tuple not in voxel_map:
                voxel_map[v_tuple] = []
            voxel_map[v_tuple].append(i)
            
        visited_voxels = set()
        clusters = []
        
        # Directions for 26-connectivity
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dx, dy, dz))
                    
        for start_voxel, pt_indices in voxel_map.items():
            if start_voxel in visited_voxels:
                continue
                
            # BFS for connected components
            queue = [start_voxel]
            visited_voxels.add(start_voxel)
            cluster_pt_indices = []
            
            while queue:
                curr_voxel = queue.pop(0)
                cluster_pt_indices.extend(voxel_map[curr_voxel])
                
                # Check neighbors
                for dx, dy, dz in directions:
                    neighbor = (curr_voxel[0] + dx, curr_voxel[1] + dy, curr_voxel[2] + dz)
                    if neighbor in voxel_map and neighbor not in visited_voxels:
                        visited_voxels.add(neighbor)
                        queue.append(neighbor)
                        
            if len(cluster_pt_indices) >= self.min_cluster_pts:
                clusters.append(points[cluster_pt_indices])
                
        return clusters
    
    def update(self, dynamic_points: np.ndarray, frame_id: int) -> List[TrackedObject]:
        """Process a new frame's dynamic points.
        1. Cluster the points
        2. Match clusters to existing tracks (greedy nearest centroid)
        3. Update matched tracks (position, velocity, bbox)
        4. Create new tracks for unmatched clusters
        5. Remove lost tracks
        Returns list of all active TrackedObject instances."""
        self.frame_count = frame_id
        
        # 1. Cluster points
        if len(dynamic_points) == 0:
            clusters = []
        elif dynamic_points.shape[1] > 3:
            clusters = self.cluster_points(dynamic_points[:, :3])
        else:
            clusters = self.cluster_points(dynamic_points)
        
        # 2. Compute centroids and bounds for new clusters
        detections = []
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            bbox_min = np.min(cluster, axis=0)
            bbox_max = np.max(cluster, axis=0)
            detections.append({
                'points': cluster,
                'centroid': centroid,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'num_points': len(cluster)
            })
            
        active_tracks = self.get_active_tracks()
        
        # Update ages
        for t in self.tracks.values():
            if self.frame_count - t.last_seen_frame <= self.max_lost:
                t.age += 1
            
        if not detections:
            return self.get_active_tracks()
            
        if not active_tracks:
            # Create new tracks for all detections
            for det in detections:
                self._create_track(det)
            return self.get_active_tracks()
            
        # Match detections to active tracks
        det_centroids = np.array([d['centroid'] for d in detections])
        trk_centroids = np.array([t.centroid for t in active_tracks])
        
        dist_matrix = cdist(det_centroids, trk_centroids)
        
        # Greedy matching
        matched_dets = set()
        matched_trks = set()
        
        while True:
            if len(matched_dets) == len(detections) or len(matched_trks) == len(active_tracks):
                break
                
            min_dist = np.inf
            best_det, best_trk = -1, -1
            
            for d in range(len(detections)):
                if d in matched_dets:
                    continue
                for t in range(len(active_tracks)):
                    if t in matched_trks:
                        continue
                    if dist_matrix[d, t] < min_dist:
                        min_dist = dist_matrix[d, t]
                        best_det = d
                        best_trk = t
                        
            if min_dist > self.max_assoc_dist:
                break
                
            matched_dets.add(best_det)
            matched_trks.add(best_trk)
            
            # Update matched track
            trk = active_tracks[best_trk]
            det = detections[best_det]
            
            # Compute velocity
            velocity = det['centroid'] - trk.centroid
            speed = np.linalg.norm(velocity) * self.assumed_fps
            
            trk.centroid = det['centroid']
            trk.velocity = velocity
            trk.speed_mps = float(speed)
            trk.bbox_min = det['bbox_min']
            trk.bbox_max = det['bbox_max']
            trk.num_points = det['num_points']
            trk.last_seen_frame = self.frame_count
            trk.hits += 1
            trk.history.append(det['centroid'])
            
        # Create new tracks for unmatched detections
        for d in range(len(detections)):
            if d not in matched_dets:
                self._create_track(detections[d])
                
        return self.get_active_tracks()
        
    def _create_track(self, det: dict) -> None:
        """Helper to initialize and register a new track."""
        trk = TrackedObject(
            track_id=self.next_id,
            centroid=det['centroid'],
            velocity=np.zeros(3),
            speed_mps=0.0,
            bbox_min=det['bbox_min'],
            bbox_max=det['bbox_max'],
            num_points=det['num_points'],
            first_seen_frame=self.frame_count,
            last_seen_frame=self.frame_count,
            age=1,
            hits=1,
            history=[det['centroid']]
        )
        self.tracks[self.next_id] = trk
        self.next_id += 1
        self._total_tracks_ever += 1
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Return all currently active (not lost) tracks."""
        return [t for t in self.tracks.values() if self.frame_count - t.last_seen_frame <= self.max_lost]
    
    def get_track_trajectories(self) -> Dict[int, List[np.ndarray]]:
        """Return {track_id: [centroid_history]} for all tracks (including lost ones recently)."""
        return {t.track_id: t.history for t in self.tracks.values()}
    
    def summary(self) -> dict:
        """Return summary stats: total_tracks_ever, active_tracks, avg_speed, max_speed."""
        active = self.get_active_tracks()
        avg_speed = np.mean([t.speed_mps for t in active]) if active else 0.0
        max_speed = np.max([t.speed_mps for t in active]) if active else 0.0
        return {
            'total_tracks_ever': self._total_tracks_ever,
            'active_tracks': len(active),
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed)
        }

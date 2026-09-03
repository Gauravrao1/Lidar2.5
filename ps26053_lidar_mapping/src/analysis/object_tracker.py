import numpy as np

from src.tracking.object_tracker import DynamicObjectTracker


class ObjectTracker:
    """Compatibility adapter for the original timestamp-based tracker API."""

    def __init__(self, distance_threshold=5.0, max_lost_frames=5):
        self._max_lost_frames = max_lost_frames
        self._tracker = DynamicObjectTracker(
            max_association_dist=distance_threshold,
            max_lost_frames=max_lost_frames,
            min_cluster_points=1,
        )

    @property
    def tracks(self):
        return self._tracker.tracks

    def update(self, points, timestamp=0.0):
        frame_id = int(round(float(timestamp)))
        tracks = self._tracker.update(np.asarray(points), frame_id=frame_id)
        self._tracker.tracks = {
            track_id: track for track_id, track in self._tracker.tracks.items()
            if frame_id - track.last_seen_frame <= self._max_lost_frames
        }
        tracks = list(self._tracker.tracks.values())
        for track in tracks:
            if track.hits == 1:
                track.velocity = None
            track.position = track.centroid
        return tracks

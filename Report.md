# Assignment 3 — Semantic Graph and Semantic Re-Localization

## System Overview

This assignment builds a semantic spatial graph from TurtleBot3 simulation data and uses it for semantic re-localization. The system collects keyframes from the robot camera, stores map-frame poses, runs object detection, generates CLIP embeddings, builds place nodes, and uses vector similarity search to estimate the robot's likely location during a second run.

## Environment

| Component | Details |
|---|---|
| Robot | TurtleBot3 simulation (`tb3_sim`) |
| ROS Distribution | ROS 2 Jazzy |
| Detector | YOLOv8n |
| Embedding Model | CLIP ViT-B/32 |
| Embedding Dimension | 512 |
| Database | PostgreSQL |
| Vector Storage/Search | pgvector |
| Semantic Graph | Apache AGE / place adjacency graph |
| Place Construction | 1m × 1m grid binning |

## Run A — Mapping Results

During Run A, the robot was moved through the maze environment to collect keyframes and build the semantic map. Keyframes were sampled based on robot movement instead of saving every camera frame, which keeps the mapping database smaller and avoids storing many duplicate frames.

| Metric | Result |
|---|---:|
| Keyframes collected | 243 |
| Places constructed | 47 |
| Grid bin size | 1m × 1m |
| Place adjacency edges | 227 |
| Detection events stored | 73 |
| Valid detections with embeddings | 2 |

The number of keyframes is higher than the number of detections because many camera frames did not contain objects that YOLOv8 could confidently classify. This is expected because the simulation world used simple geometric placeholder objects instead of realistic textured object models.

## Database Schema

The following tables were used to store the semantic mapping data:

| Table | Purpose |
|---|---|
| `detection_events` | Stores raw detection events for each accepted keyframe |
| `detections` | Stores YOLO detection results, including class, confidence, and bounding box |
| `detection_embeddings` | Stores CLIP 512-dimensional vectors using pgvector |
| `keyframes` | Stores robot pose and timestamp for each keyframe |
| `places` | Stores grid-binned spatial locations created from map-frame poses |
| `place_adjacency` | Stores edges between nearby/reachable places |
| `objects` | Stores fused object landmarks from repeated observations |

## Re-Localization (Run B)
Query: camera_view.png from robot camera

### Vector Query Results
- det_pk=3: bed, conf=0.27, distance=0.0888
- det_pk=4: traffic light, conf=0.34, distance=0.1253

### Top-3 Candidate Places
1. Place 30: (4.0, 2.0) — score=1.7859
   - Reachable within 2 hops: 14 places

### Best Pose Hypothesis
- x=3.64, y=2.43, yaw=-0.19

## Success Analysis
- Vector KNN search successfully found visually similar frames
- Graph traversal correctly identified reachable places
- Pose hypothesis generated from best matching place

## Reproducibility
1. docker compose up demo-world
2. python3 semantic_mapper.py  (Run A - mapping)
3. Navigate robot around maze using RViz2 2D Nav Goal
4. python3 build_graph.py
5. python3 relocalizer.py  (Run B - re-localization)

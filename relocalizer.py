import psycopg2
import torch
import cv2
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage


DB_CONFIG = {
    "host": "localhost",
    "dbname": "robotics",
    "user": "turtlebot",
    "password": "turtlebot",
}

QUERY_IMAGE_PATH = "/workspaces/turtlebot-maze/camera_view.png"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K_DETECTIONS = 5
TOP_K_PLACES = 3
GRAPH_HOPS = 2


def load_clip():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    print("CLIP model loaded")
    return model, processor


def get_embedding(model, processor, image_rgb):
    pil_image = PILImage.fromarray(image_rgb)

    inputs = processor(
        images=pil_image,
        return_tensors="pt",
    )

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features[0].cpu().numpy().astype(float).tolist()


def vector_query(cur, embedding, k=5):
    cur.execute(
        """
        SELECT
            de.det_pk,
            d.class_name,
            d.confidence,
            de.embedding <=> %s::vector AS distance,
            de.model
        FROM detection_embeddings de
        JOIN detections d
            ON de.det_pk = d.det_pk
        ORDER BY de.embedding <=> %s::vector
        LIMIT %s
        """,
        (
            str(embedding),
            str(embedding),
            k,
        ),
    )

    return cur.fetchall()


def get_place_for_detection(cur, det_pk):
    cur.execute(
        """
        SELECT
            k.place_id,
            k.map_x,
            k.map_y,
            k.map_yaw
        FROM detections d
        JOIN detection_events e
            ON d.event_id = e.event_id
        JOIN keyframes k
            ON e.run_id = k.run_id
           AND ABS(EXTRACT(EPOCH FROM (k.stamp - e.stamp))) < 1.0
        WHERE d.det_pk = %s
        ORDER BY ABS(EXTRACT(EPOCH FROM (k.stamp - e.stamp)))
        LIMIT 1
        """,
        (det_pk,),
    )

    return cur.fetchone()


def graph_query(cur, start_place_id, hops=2):
    visited = set()
    frontier = {start_place_id}

    for _ in range(hops):
        if not frontier:
            break

        placeholders = ",".join(["%s"] * len(frontier))

        cur.execute(
            f"""
            SELECT place_b
            FROM place_adjacency
            WHERE place_a IN ({placeholders})
            """,
            list(frontier),
        )

        next_places = {row[0] for row in cur.fetchall()}

        cur.execute(
            f"""
            SELECT place_a
            FROM place_adjacency
            WHERE place_b IN ({placeholders})
            """,
            list(frontier),
        )

        next_places.update({row[0] for row in cur.fetchall()})

        visited.update(frontier)
        frontier = next_places - visited

    visited.update(frontier)
    return visited


def rank_candidate_places(cur, vector_results):
    place_scores = {}

    for det_pk, class_name, confidence, distance, model_name in vector_results:
        place_row = get_place_for_detection(cur, det_pk)

        if not place_row:
            continue

        place_id, map_x, map_y, map_yaw = place_row

        similarity_score = max(0.0, 1.0 - float(distance))
        confidence_score = float(confidence)
        final_score = similarity_score + (0.25 * confidence_score)

        if place_id not in place_scores:
            place_scores[place_id] = {
                "score": 0.0,
                "matches": [],
                "pose": (map_x, map_y, map_yaw),
            }

        place_scores[place_id]["score"] += final_score
        place_scores[place_id]["matches"].append(
            {
                "det_pk": det_pk,
                "class_name": class_name,
                "confidence": confidence,
                "distance": distance,
                "model": model_name,
            }
        )

    ranked = sorted(
        place_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    return ranked[:TOP_K_PLACES]


def print_vector_results(results):
    print("\nVector Query Results")
    print("--------------------")

    if not results:
        print("No stored embeddings were found in the database.")
        return

    for rank, row in enumerate(results, start=1):
        det_pk, class_name, confidence, distance, model_name = row

        print(
            f"{rank}. det_pk={det_pk} | "
            f"class={class_name} | "
            f"conf={confidence:.2f} | "
            f"distance={distance:.4f} | "
            f"model={model_name}"
        )


def print_candidate_places(cur, ranked_places):
    print("\nTop Candidate Places")
    print("--------------------")

    if not ranked_places:
        print("No candidate places were found.")
        return

    for rank, (place_id, info) in enumerate(ranked_places, start=1):
        cur.execute(
            """
            SELECT center_x, center_y, label
            FROM places
            WHERE place_id = %s
            """,
            (place_id,),
        )

        place = cur.fetchone()

        if place:
            center_x, center_y, label = place
        else:
            center_x, center_y, label = None, None, "unknown"

        reachable = graph_query(cur, place_id, hops=GRAPH_HOPS)

        print(
            f"{rank}. Place {place_id} | "
            f"center=({center_x:.1f}, {center_y:.1f}) | "
            f"label={label} | "
            f"score={info['score']:.4f}"
        )

        print(f"   Reachable places within {GRAPH_HOPS} hops: {len(reachable)}")

        for match in info["matches"]:
            print(
                f"   Match det_pk={match['det_pk']} | "
                f"class={match['class_name']} | "
                f"conf={match['confidence']:.2f} | "
                f"distance={match['distance']:.4f}"
            )


def print_best_pose(ranked_places):
    print("\nBest Pose Hypothesis")
    print("--------------------")

    if not ranked_places:
        print("No pose hypothesis could be generated.")
        return

    best_place_id, best_info = ranked_places[0]
    x, y, yaw = best_info["pose"]

    print(f"Best place: Place {best_place_id}")
    print(f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")


def load_query_image(path):
    image_bgr = cv2.imread(path)

    if image_bgr is None:
        print(f"Query image not found at: {path}")
        print("Using random test image only to verify that the script runs.")
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def relocalize(query_image_rgb):
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        cur = conn.cursor()

        model, processor = load_clip()

        print("\nRunning Semantic Re-Localization")
        print("--------------------------------")

        query_embedding = get_embedding(
            model,
            processor,
            query_image_rgb,
        )

        vector_results = vector_query(
            cur,
            query_embedding,
            k=TOP_K_DETECTIONS,
        )

        print_vector_results(vector_results)

        ranked_places = rank_candidate_places(
            cur,
            vector_results,
        )

        print_candidate_places(
            cur,
            ranked_places,
        )

        print_best_pose(ranked_places)

    finally:
        conn.close()


if __name__ == "__main__":
    query_image = load_query_image(QUERY_IMAGE_PATH)
    relocalize(query_image)

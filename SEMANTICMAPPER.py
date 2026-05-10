import uuid
import hashlib
from math import atan2, sqrt
from datetime import datetime, timezone

import cv2
import torch
import rclpy
import psycopg2
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel


RUN_ID = str(uuid.uuid4())
ROBOT_ID = "tb3_sim"

DB_CONFIG = {
    "host": "localhost",
    "dbname": "robotics",
    "user": "turtlebot",
    "password": "turtlebot",
}

IMAGE_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.25

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

KEYFRAME_DIST = 0.30
KEYFRAME_ANGLE = 0.20
GRID_SIZE = 1.0


class SemanticMapper(Node):
    def __init__(self):
        super().__init__("semantic_mapper")

        self.bridge = CvBridge()
        self.latest_odom = None

        self.last_kf_x = None
        self.last_kf_y = None
        self.last_kf_yaw = None

        self.keyframe_count = 0
        self.sequence = 0

        self.get_logger().info("Loading YOLOv8n detector...")
        self.yolo = YOLO(YOLO_MODEL)

        self.get_logger().info("Loading CLIP embedding model...")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model.eval()

        self.conn = psycopg2.connect(**DB_CONFIG)

        self.odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            2,
        )

        self.get_logger().info("Semantic Mapper started")
        self.get_logger().info(f"Robot ID: {ROBOT_ID}")
        self.get_logger().info(f"Run ID: {RUN_ID}")

    def get_conn(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def odom_callback(self, msg):
        self.latest_odom = msg

    def get_yaw(self, odom_msg):
        q = odom_msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    def is_keyframe(self, x, y, yaw):
        if self.last_kf_x is None:
            return True

        dist = sqrt((x - self.last_kf_x) ** 2 + (y - self.last_kf_y) ** 2)
        angle_diff = abs(yaw - self.last_kf_yaw)

        return dist >= KEYFRAME_DIST or angle_diff >= KEYFRAME_ANGLE

    def update_last_keyframe_pose(self, x, y, yaw):
        self.last_kf_x = x
        self.last_kf_y = y
        self.last_kf_yaw = yaw

    def get_or_create_place(self, cur, run_id, x, y):
        grid_x = round(x / GRID_SIZE) * GRID_SIZE
        grid_y = round(y / GRID_SIZE) * GRID_SIZE

        cur.execute(
            """
            SELECT place_id
            FROM places
            WHERE run_id = %s
              AND center_x = %s
              AND center_y = %s
            """,
            (run_id, grid_x, grid_y),
        )

        row = cur.fetchone()
        if row:
            return row[0]

        label = f"place_{grid_x:.1f}_{grid_y:.1f}"

        cur.execute(
            """
            INSERT INTO places
            (run_id, center_x, center_y, label)
            VALUES (%s, %s, %s, %s)
            RETURNING place_id
            """,
            (run_id, grid_x, grid_y, label),
        )

        return cur.fetchone()[0]

    def compute_clip_embedding(self, crop_rgb):
        pil_image = PILImage.fromarray(crop_rgb)

        inputs = self.clip_processor(
            images=pil_image,
            return_tensors="pt",
        )

        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features[0].cpu().numpy().astype(float).tolist()

    def insert_detection_event(self, cur, msg, event_id, stamp, x, y, yaw):
        image_bytes = bytes(msg.data)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        cur.execute(
            """
            INSERT INTO detection_events
            (
                event_id,
                run_id,
                robot_id,
                sequence,
                stamp,
                image_frame_id,
                image_sha256,
                width,
                height,
                encoding,
                x,
                y,
                yaw,
                vx,
                vy,
                wz,
                tf_ok,
                t_base_camera,
                raw_event
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                event_id,
                RUN_ID,
                ROBOT_ID,
                self.sequence,
                stamp,
                msg.header.frame_id,
                image_sha256,
                msg.width,
                msg.height,
                msg.encoding,
                x,
                y,
                yaw,
                0.0,
                0.0,
                0.0,
                False,
                [0.0] * 16,
                "{}",
            ),
        )

    def insert_keyframe(self, cur, stamp, x, y, yaw, place_id):
        cur.execute(
            """
            INSERT INTO keyframes
            (
                run_id,
                robot_id,
                stamp,
                map_x,
                map_y,
                map_yaw,
                place_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING keyframe_id
            """,
            (
                RUN_ID,
                ROBOT_ID,
                stamp,
                x,
                y,
                yaw,
                place_id,
            ),
        )

        return cur.fetchone()[0]

    def insert_or_update_object(self, cur, cls_name, x, y, place_id, stamp):
        cur.execute(
            """
            SELECT object_id
            FROM objects
            WHERE run_id = %s
              AND class_name = %s
              AND sqrt(power(mean_x - %s, 2) + power(mean_y - %s, 2)) < 1.0
            ORDER BY sqrt(power(mean_x - %s, 2) + power(mean_y - %s, 2))
            LIMIT 1
            """,
            (
                RUN_ID,
                cls_name,
                x,
                y,
                x,
                y,
            ),
        )

        row = cur.fetchone()

        if row:
            object_id = row[0]

            cur.execute(
                """
                UPDATE objects
                SET
                    mean_x = (mean_x * observation_count + %s) / (observation_count + 1),
                    mean_y = (mean_y * observation_count + %s) / (observation_count + 1),
                    observation_count = observation_count + 1,
                    last_seen = %s
                WHERE object_id = %s
                """,
                (
                    x,
                    y,
                    stamp,
                    object_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO objects
                (
                    run_id,
                    class_name,
                    mean_x,
                    mean_y,
                    place_id,
                    first_seen,
                    last_seen
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    RUN_ID,
                    cls_name,
                    x,
                    y,
                    place_id,
                    stamp,
                    stamp,
                ),
            )

    def process_detections(self, cur, results, event_id, cv_image, x, y, place_id, stamp):
        det_count = 0
        embedding_count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                det_id = str(uuid.uuid4())

                cur.execute(
                    """
                    INSERT INTO detections
                    (
                        event_id,
                        det_id,
                        class_id,
                        class_name,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING det_pk
                    """,
                    (
                        event_id,
                        det_id,
                        cls_id,
                        cls_name,
                        conf,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                    ),
                )

                row = cur.fetchone()

                if not row:
                    continue

                det_pk = row[0]

                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(cv_image.shape[1], int(bbox[2]))
                y2 = min(cv_image.shape[0], int(bbox[3]))

                if x2 > x1 and y2 > y1:
                    crop = cv_image[y1:y2, x1:x2]
                    embedding = self.compute_clip_embedding(crop)

                    cur.execute(
                        """
                        INSERT INTO detection_embeddings
                        (
                            det_pk,
                            model,
                            embedding
                        )
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            det_pk,
                            "clip-vit-base-patch32",
                            str(embedding),
                        ),
                    )

                    embedding_count += 1

                self.insert_or_update_object(
                    cur,
                    cls_name,
                    x,
                    y,
                    place_id,
                    stamp,
                )

                det_count += 1

        return det_count, embedding_count

    def image_callback(self, msg):
        if self.latest_odom is None:
            return

        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        yaw = self.get_yaw(self.latest_odom)

        if not self.is_keyframe(x, y, yaw):
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        self.sequence += 1
        self.keyframe_count += 1
        self.update_last_keyframe_pose(x, y, yaw)

        stamp = datetime.fromtimestamp(
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            tz=timezone.utc,
        )

        event_id = str(uuid.uuid4())
        zenoh_key = f"maze/{ROBOT_ID}/{RUN_ID}/detections/v1/{event_id}"

        results = self.yolo.predict(
            cv_image,
            conf=YOLO_CONF,
            verbose=False,
        )

        conn = self.get_conn()

        try:
            with conn.cursor() as cur:
                self.insert_detection_event(
                    cur,
                    msg,
                    event_id,
                    stamp,
                    x,
                    y,
                    yaw,
                )

                place_id = self.get_or_create_place(
                    cur,
                    RUN_ID,
                    x,
                    y,
                )

                keyframe_id = self.insert_keyframe(
                    cur,
                    stamp,
                    x,
                    y,
                    yaw,
                    place_id,
                )

                det_count, embedding_count = self.process_detections(
                    cur,
                    results,
                    event_id,
                    cv_image,
                    x,
                    y,
                    place_id,
                    stamp,
                )

            conn.commit()

            self.get_logger().info(
                f"KF={self.keyframe_count} "
                f"seq={self.sequence} "
                f"keyframe_id={keyframe_id} "
                f"place={place_id} "
                f"pose=({x:.2f}, {y:.2f}, {yaw:.2f}) "
                f"detections={det_count} "
                f"embeddings={embedding_count}"
            )

            self.get_logger().info(f"event_id={event_id}")
            self.get_logger().info(f"zenoh_key={zenoh_key}")

        except Exception as e:
            conn.rollback()
            self.get_logger().error(f"Database insert failed: {e}")


def main():
    rclpy.init()
    node = SemanticMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

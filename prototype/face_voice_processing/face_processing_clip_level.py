import os
import json
import base64
from threading import Lock
import cv2
import hdbscan
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import argparse

from prototype.tools.utils import extract_keyframes_from_fact

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
face_app_lock = Lock()

global_face_id_counter = 0
face_id_lock = Lock()

def reset_face_id_counter():
    global global_face_id_counter
    with face_id_lock:
        global_face_id_counter = 0
    print(f"  🔄 Face ID counter reset to 0")

def get_next_face_id():
    global global_face_id_counter
    with face_id_lock:
        face_id = f"face_{global_face_id_counter}"
        global_face_id_counter += 1
    return face_id

def collect_clip_inputs(clip_data: dict):
    frames_with_meta = []
    for fact in clip_data.get("facts", []):
        frames = extract_keyframes_from_fact(fact)
        frames_with_meta.extend(frames)
    return frames_with_meta

def extract_faces_from_base64(timestamp, jpg_path, base64_img, fact_id, clip_name):

    img = None

    if jpg_path and os.path.exists(jpg_path):
        img = cv2.imread(jpg_path)

    if img is None:
        return []

    try:
        with face_app_lock:
            detected_faces = face_app.get(img)
    except Exception as e:
        print(f"  ❌ InsightFace Error for {fact_id}: {type(e).__name__}: {e}")
        return []

    faces = []
    for face in detected_faces:
        bbox = [int(x) for x in face.bbox.astype(int).tolist()]
        emb = face.normed_embedding.tolist()
        faces.append({
            "fact_id": fact_id,
            "timestamp": timestamp,
            "clip_name": clip_name,
            "bounding_box": bbox,
            "face_emb": emb,
            "jpg_path": jpg_path,
            "base64_img": base64_img if isinstance(base64_img, str) else "",
        })
    return faces

def extract_faces_batch(frames, clip_name, num_workers=4):
    def process_one(frame):
        return extract_faces_from_base64(frame["timestamp"], frame["jpg_path"], frame["base64_frame"], frame["fact_id"], clip_name)

    all_faces = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for faces in tqdm(
            executor.map(process_one, frames),
            total=len(frames),
            desc=f"Extracting faces for clip {clip_name}"
        ):
            all_faces.extend(faces)
    return all_faces

def compute_centroids(persons):

    return {
        pid: np.mean([np.array(f["face_emb"]) for f in cluster], axis=0)
        for pid, cluster in persons.items()
    }

def cluster_faces_initial_strict(faces, min_cluster_size=2, sim_threshold=0.15):

    if not faces:
        return {}, {}

    if len(faces) < min_cluster_size:
        print(f"  ℹ️  Only {len(faces)} face(s) detected, creating single cluster")
        temp_id = "temp_0"
        return {temp_id: faces}, {temp_id: np.mean([np.array(f["face_emb"]) for f in faces], axis=0)}

    embeddings = np.array([f["face_emb"] for f in faces])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(embeddings)

    persons = {}
    cluster_to_id = {}
    noise_faces = []

    for i, label in enumerate(labels):
        if label == -1:
            noise_faces.append((i, faces[i]))
        else:
            if label not in cluster_to_id:
                cluster_to_id[label] = f"temp_{label}"
            temp_id = cluster_to_id[label]
            persons.setdefault(temp_id, []).append(faces[i])

    if noise_faces:
        if not persons:
            for idx, (i, face) in enumerate(noise_faces):
                temp_id = f"temp_noise_{idx}"
                persons[temp_id] = [face]
        else:
            centroids = compute_centroids(persons)
            for idx, face in noise_faces:
                emb = np.array(face["face_emb"])
                sims = {
                    pid: cosine_similarity([emb], [cent])[0, 0]
                    for pid, cent in centroids.items()
                }
                best_pid, best_sim = max(sims.items(), key=lambda x: x[1])

                if best_sim >= sim_threshold:
                    persons[best_pid].append(face)
                else:
                    temp_id = f"temp_noise_{idx}"
                    persons[temp_id] = [face]

    return persons, compute_centroids(persons)


def merge_clip_clusters_to_global(
    clip_persons,
    clip_centroids,
    existing_persons,
    existing_centroids,
    sim_threshold=0.15
):

    if not existing_persons:
        global_persons = {}
        global_centroids = {}
        for temp_id, faces in clip_persons.items():
            global_id = get_next_face_id()
            global_persons[global_id] = faces
            global_centroids[global_id] = clip_centroids[temp_id]
            print(f"  ✨ Created {global_id} from {temp_id}")
        return global_persons, global_centroids

    for temp_id, faces in clip_persons.items():
        clip_center = clip_centroids[temp_id]

        sims = {
            pid: cosine_similarity(
                clip_center.reshape(1, -1),
                global_centroid.reshape(1, -1)
            )[0, 0]
            for pid, global_centroid in existing_centroids.items()
        }

        best_pid, best_sim = max(sims.items(), key=lambda x: x[1])

        if best_sim >= sim_threshold:
            existing_persons[best_pid].extend(faces)
            print(f"  🔗 Merged {temp_id} to existing {best_pid} (sim={best_sim:.2f})")
        else:
            new_pid = get_next_face_id()
            existing_persons[new_pid] = faces
            print(f"  ✨ Created new {new_pid} from {temp_id}")

    existing_centroids = compute_centroids(existing_persons)
    return existing_persons, existing_centroids

def save_faces_images(persons, faces_dir, video_name):

    for pid, faces in persons.items():
        for idx, face in enumerate(faces):
            clip_name = face["clip_name"]
            person_dir = os.path.join(faces_dir, video_name, str(clip_name), pid)
            os.makedirs(person_dir, exist_ok=True)

            try:
                img = cv2.imread(face["jpg_path"])

                if img is None:
                    continue

                x1, y1, x2, y2 = face["bounding_box"]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                cropped = img[y1:y2, x1:x2]

                out_path = os.path.join(person_dir, f"frame_{face['timestamp']}_{idx}.jpg")
                cv2.imwrite(out_path, cropped)
                face["image_path"] = out_path

                _, jpg_buffer = cv2.imencode(".jpg", cropped)
                face["image_base64"] = base64.b64encode(jpg_buffer).decode("utf-8")

            except Exception as e:
                print(f"[⚠️] Failed to save / encode face: {e}")
                continue

    return persons

def set_face_id_counter(val):
    global global_face_id_counter
    with face_id_lock:
        global_face_id_counter = val

def process_clip_faces_online(
    clip_id,
    clip_data,
    video_name,
    faces_dir,
    faces_images_dir,
    existing_persons,
    existing_centroids,
    sim_threshold=0.15,
    reset_counter=False,
):

    if reset_counter:
        reset_face_id_counter()

    frames = collect_clip_inputs(clip_data)
    if not frames:
        print(f"[Face] Clip {clip_id}: no keyframes found")
        return existing_persons, existing_centroids

    batch_faces = extract_faces_batch(frames, clip_id)
    if not batch_faces:
        print(f"[Face] Clip {clip_id}: no faces detected")
        return existing_persons, existing_centroids

    clip_persons, clip_centroids = cluster_faces_initial_strict(batch_faces, min_cluster_size=2)
    if not clip_persons:
        print(f"[Face] Clip {clip_id}: clustering failed, using virtual faces from keyframes")
        return existing_persons, existing_centroids

    existing_persons, existing_centroids = merge_clip_clusters_to_global(
       clip_persons,
        clip_centroids,
        existing_persons,
        existing_centroids,
        sim_threshold=sim_threshold,
    )

    existing_persons = save_faces_images(existing_persons, faces_images_dir, video_name)

    for pid, faces in existing_persons.items():
        output_file = os.path.join(faces_dir, f"{pid}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"face_id": pid, "faces": faces}, f, ensure_ascii=False, indent=2)

    print(f"[Face] Clip {clip_id}: merged, now total {len(existing_persons)} persons")
    return existing_persons, existing_centroids

def parse_arguments():
    parser = argparse.ArgumentParser(description="Incremental face clustering from facts JSON (M3-Agent style online merging).")
    parser.add_argument("--facts", default='../test_with_timestamp/_cZXyj6rYVg.json', help="Path to the input facts JSON file.")
    parser.add_argument("--out-character", default="../faces", help="Output folder for per-person JSONs.")
    parser.add_argument("--out-faces", default="../faces_images", help="Output folder for extracted face crops.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for face extraction (currently not used explicitly).")
    parser.add_argument("--sim-threshold", type=float, default=0.45, help="Cosine similarity threshold for merging clip clusters to global persons.")
    return parser.parse_args()

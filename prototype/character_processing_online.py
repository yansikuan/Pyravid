import os
from prototype.face_voice_processing.face_processing_clip_level import process_clip_faces_online
from prototype.face_voice_processing.face_voice_alignment import process_clip_equivalence_online
from prototype.face_voice_processing.face_voice_caption_generation import process_clip_captions_online
from prototype.face_voice_processing.merge_face_voice_alignment import process_clip
from prototype.face_voice_processing.voice_processing import process_clip_voices_online
from prototype.tools.vectorstore import Qdrant
from prototype.tools.utils import load_facts
from prototype.face_voice_processing.face_processing_clip_level import set_face_id_counter
import pickle
import argparse

db = None

def save_state(state_path, existing_persons, existing_centroids):
    with open(state_path, "wb") as f:
        pickle.dump({
            "existing_persons": existing_persons,
            "existing_centroids": existing_centroids
        }, f)

def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        existing_persons = state.get("existing_persons", {})
        existing_centroids = state.get("existing_centroids", {})
        max_id = 0
        for pid in existing_persons.keys():
            try:
                num = int(str(pid).replace("face_", ""))
                if num >= max_id:
                    max_id = num + 1
            except Exception:
                continue
        set_face_id_counter(max_id)
        return existing_persons, existing_centroids
    else:
        return {}, {}

def initialize_db(video_name, work_dir="./results"):
    global db
    db = Qdrant(
        collection_name=f"character",
        embedding_model_dims=3072,
        snapshot_dir=f"{work_dir}/snapshot/{video_name}",
        path=f"../character_qdrant_rebuttal/qdrant_{video_name}",
        on_disk=True,
    )
    print("  🔄 Database initialized")

def process_characters_clip_online(video_folder, facts_path, work_dir="./results"):
    try:
        faces_dir = f"{work_dir}/faces"
        faces_images_dir = f"{work_dir}/faces_images"
        voices_dir = f"{work_dir}/voices"
        captions_dir = f"{work_dir}/face_voice_captions"
        equivalence_dir = f"{work_dir}/equivalence"
        character_dir = f"{work_dir}/character"
        merged_facts_dir = f"{work_dir}/facts_with_featureID"
        id_mapping_dir= f"{work_dir}/global_id_mapping"
        snapshot_dir = f"{work_dir}/snapshot"

        for d in [
            faces_dir,
            faces_images_dir,
            voices_dir,
            captions_dir,
            equivalence_dir,
            character_dir,
            merged_facts_dir,
            id_mapping_dir,
            snapshot_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        facts_data = load_facts(facts_path)
        video_name = os.path.splitext(os.path.basename(facts_path))[0]
        state_path = f"{work_dir}/clip_state_{video_name}.pkl"
        faces_dir = f"{faces_dir}/{video_name}"

        existing_persons, existing_centroids= load_state(state_path)

        character_dir = f"{character_dir}/{video_name}"
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(character_dir, exist_ok=True)

        captions_path = captions_dir

        print(f"[🚀] Start clip-level online processing for video: {video_name}")

        initialize_db(video_name=video_name, work_dir=work_dir)
        print(f"  🔄 Database initialized for video: {video_name}")
        for clip_id, clip_data in facts_data.items():

            print("\n============================")
            print(f"[🎬] Processing clip {clip_id} (online)")
            print("============================")

            print(f" Starting face processing with existing persons: {len(existing_persons)}")
            # [1] faces
            existing_persons, existing_centroids = process_clip_faces_online(
                clip_id=clip_id,
                clip_data=clip_data,
                video_name=video_name,
                faces_dir=faces_dir,
                faces_images_dir=faces_images_dir,
                existing_persons=existing_persons,
                existing_centroids=existing_centroids,
                sim_threshold=0.45,
            )

            # existing_persons = process_clip_faces_online(
            #     clip_id=clip_id,
            #     clip_data=clip_data,
            #     video_name=video_name,
            #     faces_dir=faces_dir,
            #     faces_images_dir=faces_images_dir,
            #     existing_persons=existing_persons,
            #     #existing_centroids=existing_centroids,
            #     #sim_threshold=0.45,
            # )

            # [2] voices
            process_clip_voices_online(
               clip_id=clip_id,
               clip_data=clip_data,
               video_folder=video_folder,
               voices_dir=voices_dir,
               video_name=video_name,
               sample_rate=16000,
           )

            # [3] captions
            process_clip_captions_online(
               clip_id=clip_id,
               clip_data=clip_data,
               facts_path=facts_path,
               faces_dir=faces_dir,
               voices_dir=voices_dir,
               captions_dir=captions_dir,
               video_folder=video_folder,
           )

            # [4] equivalence
            process_clip_equivalence_online(
               clip_id=clip_id,
               captions_path=f"{captions_path}/{video_name}/{clip_id}.json",   # now a directory
               faces_dir=faces_dir,
               voices_dir=voices_dir,
               equivalence_dir=equivalence_dir
           )

            # [5] incremental merge
            process_clip(clip_id = clip_id,
                        faces_dir = faces_dir,
                        voices_dir = f"{voices_dir}/{video_name}/{clip_id}",
                        equivalence_path = f"{equivalence_dir}/{video_name}/{clip_id}.json",
                        character_dir = character_dir,
                       caption_file_path = f"{captions_dir}/{video_name}/{clip_id}.json",
                       snapshot_dir = f"{snapshot_dir}/{video_name}",
                       facts_path = facts_path,
                       facts_with_featureID_path = f"{merged_facts_dir}/{video_name}.json",
                       db = db)

            save_state(state_path, existing_persons, existing_centroids)
            print(f"  💾 State saved after processing clip {clip_id}")

        print("\n✔ Clip-level ONLINE character processing completed.")

    except Exception as e:
        print(f"[⚠️] Character processing failed: {e}")
        return []

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clip-level Online Character Processing"
    )

    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help="Path to video clips folder"
    )

    parser.add_argument(
        "--facts_path",
        type=str,
        required=True,
        help="Path to facts json file"
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="./results",
        help="Working directory for outputs"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("\n========== CONFIG ==========")
    print(f"Video Folder: {args.video_folder}")
    print(f"Facts Path  : {args.facts_path}")
    print(f"Work Dir    : {args.work_dir}")
    print("============================\n")

    process_characters_clip_online(
        video_folder=args.video_folder,
        facts_path=args.facts_path,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
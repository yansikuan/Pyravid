import os
import glob
import re
import uuid
from typing import Dict, List, Tuple, Set

from prototype.tools.prompts import CHARACTER_PROFILE_SUMMARIZATION
from prototype.tools.utils import (
    call_model,
    load_facts,
    smart_json_loads,
    save_json,
)

def replace_all_ids(text: str, id_mapping: Dict[str, str]) -> str:

    if not text:
        return text

    def replace_match(match):
        entity_id = match.group(1)

        if entity_id in id_mapping:
            return f"<{id_mapping[entity_id]}>"
        return match.group(0)


    text = re.sub(r'<([^>]+)>', replace_match, text)

    return text
def load_all_faces(faces_clip_dir: str) -> Dict[str, Dict]:
    face_index: Dict[str, Dict] = {}
    if os.path.exists(faces_clip_dir):
        for ff in glob.glob(os.path.join(faces_clip_dir, "*.json")):
            data = load_facts(ff)
            face_id = data.get("face_id")
            if face_id:
                face_index[face_id] = data
    print(f"  👤 Loaded {len(face_index)} faces")
    return face_index

def build_id_mapping(face_index: Dict[str, Dict], voice_index: Dict[str, Dict],
                     equivalence_links: List[Tuple[str, str, str]]) -> Dict[str, str]:
    id_mapping: Dict[str, str] = {}

    for face_id in face_index.keys():
        person_id = face_id.replace("face_", "person_")
        id_mapping[face_id] = person_id

    for face_id, _, voice_id in equivalence_links:
        if face_id in id_mapping:
            person_id = id_mapping[face_id]
            id_mapping[voice_id] = person_id

    print(f"  🔄 Built ID mapping: {len(id_mapping)} entities")
    return id_mapping

def load_existing_persons(character_dir: str) -> Dict[str, Dict]:
    existing_person_data: Dict[str, Dict] = {}
    for person_file in glob.glob(os.path.join(character_dir, "person_*.json")):
        try:
            person_obj = load_facts(person_file)
            person_id = person_obj.get("person_id")
            if person_id:
                existing_person_data[person_id] = person_obj
                print(f"  📂 Found existing person: {person_id}")
        except Exception as e:
            print(f"  ⚠️ Failed to load {person_file}: {e}")
    return existing_person_data

def create_or_update_person(face_id: str,
                           existing_persons: Dict[str, Dict]) -> Dict:
    person_id = face_id.replace("face_", "person_")

    if person_id in existing_persons:
        person_obj = existing_persons[person_id]
        print(f"  🔄 Updating existing person: {person_id}")
    else:
        person_obj = {
            "person_id": person_id,
            "face_id": face_id,
            "voice_ids": [],
            "facts": [],
            "character_profile": {"description": ""} ,
        }
        print(f"  ✨ Created new person: {person_id} from {face_id}")

    return person_obj

def load_equivalence_links(equivalence_path: str, clip_id: str) -> List[Tuple[str, str, str]]:
    current_links: List[Tuple[str, str, str]] = []
    if os.path.exists(equivalence_path):
        eq_data = load_facts(equivalence_path)
        block = eq_data.get(str(clip_id), {})
        for fact_id, fact_block in (block.items() if isinstance(block, dict) else []):
            for pair in fact_block.get("Equivalence", []):
                if isinstance(pair, list) and len(pair) == 2:
                    entity_0 = pair[0].strip("<>")
                    entity_1 = pair[1].strip("<>")

                    if entity_0.startswith("face_") and entity_1.startswith("voice_"):
                        face_id = entity_0
                        voice_id = entity_1
                    elif entity_1.startswith("face_") and entity_0.startswith("voice_"):
                        face_id = entity_1
                        voice_id = entity_0
                    else:
                        continue

                    current_links.append((face_id, fact_id, voice_id))

    print(f"  🧩 Loaded {len(current_links)} equivalence links")
    return current_links

def attach_voices_to_person(person_obj: Dict, equivalence_links: List[Tuple[str, str, str]],
                           voice_index: Dict[str, Dict]) -> bool:
    face_id = person_obj["face_id"]
    initial_voice_count = len(person_obj.get("voice_ids", []))

    for link_face_id, fact_id, voice_id in equivalence_links:
        if link_face_id == face_id and voice_id in voice_index:
            if voice_id not in person_obj["voice_ids"]:
                person_obj["voice_ids"].append(voice_id)
                print(f"    🔗 Linked voice {voice_id} to {person_obj['person_id']}")

    new_voice_count = len(person_obj["voice_ids"])
    return new_voice_count > initial_voice_count

def load_facts_captions(caption_file_path: str, clip_id: str) -> Dict[str, str]:
    facts_captions: Dict[str, str] = {}

    if not os.path.exists(caption_file_path):
        print(f"  ⚠️ Caption file not found: {caption_file_path}")
        return facts_captions

    try:
        caption_data = load_facts(caption_file_path)
        clip_data = caption_data.get(str(clip_id), {})

        model_out = clip_data.get("model_output", {})
        summary = model_out.get("character_level_summary", "")
        facts = model_out.get("facts", {})

        for fact_id, fact_content in facts.items():
            if isinstance(fact_content, dict):
               facts_captions[fact_id] = {
                    "character_level_summary": summary,
                    "character_level_facts": fact_content.get("character_level_facts", ""),
                    "character_details": fact_content.get("character_details", {})
                }
        print(f"  📄 Loaded {len(facts_captions)} character-level facts from caption")
    except Exception as e:
        print(f"  ❌ Failed to load caption file: {e}")

    return facts_captions

def collect_facts_for_person_v1(person_obj: Dict, face_index: Dict[str, Dict],
                             voice_index: Dict[str, Dict], facts_captions: Dict[str, Dict], clip_id: str) -> bool:

    existing_facts_dict = {f['fact_id']: f for f in person_obj.get('facts', [])}
    old_fact_ids: Set[str] = set(existing_facts_dict.keys())

    person_id = person_obj["person_id"]
    face_id = person_obj["face_id"]
    voice_ids = person_obj.get("voice_ids", [])

    if face_id in face_index:
        face_data = face_index[face_id]
        for face_info in face_data.get("faces", []):
            if "fact_id" in face_info:
                fact_id = face_info.get("fact_id", "")
                clip_name = face_info.get("clip_name", "")

                if fact_id in facts_captions and fact_id not in existing_facts_dict:
                    fact_data = facts_captions[fact_id]
                    existing_facts_dict[fact_id] = {
                        "fact_id": fact_id,
                        "clip_id": clip_name,
                        "character_level_facts": fact_data.get("character_level_facts", ""),
                        "character_details": fact_data.get("character_details", {}),
                        "source": "face"
                    }

    for voice_id in voice_ids:
        if voice_id in voice_index:
            voice_data = voice_index[voice_id]
            if "fact_id" in voice_data:
                clip_name= voice_data.get("clip_name", "")
                fact_id = voice_data["fact_id"]

                if fact_id in facts_captions and fact_id not in existing_facts_dict:
                    fact_data = facts_captions[fact_id]
                    existing_facts_dict[fact_id] = {
                        "fact_id": fact_id,
                        "clip_id": clip_name,
                        "character_level_facts": fact_data.get("character_level_facts", ""),
                        "character_details": fact_data.get("character_details", {}),
                        "source": "voice"
                    }

    for fact_id, fact_data in facts_captions.items():
        if fact_id in existing_facts_dict:
            continue

        character_level_facts = fact_data.get("character_level_facts", "")
        if f"<{person_id}>" in character_level_facts:
            existing_facts_dict[fact_id] = {
                "fact_id": fact_id,
                "clip_id": clip_id,
                "character_level_facts": fact_data.get("character_level_facts", ""),
                "character_details": fact_data.get("character_details", {}),
                "source": "mentioned"
            }

            print(f"    🔗 {person_id} mentioned in {fact_id}, adding to facts")

    person_obj['facts'] = list(existing_facts_dict.values())

    all_fact_ids = set(existing_facts_dict.keys())
    newly_added = all_fact_ids - old_fact_ids

    has_changes = len(newly_added) > 0
    if has_changes:
        print(f"    🆕 Added {len(newly_added)} new facts for {person_obj['person_id']}: {len(old_fact_ids)} -> {len(all_fact_ids)}")
    else:
        print(f"    ℹ️ No new facts for {person_obj['person_id']} (total: {len(all_fact_ids)})")

    return has_changes

def collect_facts_for_person(person_obj: Dict, face_index: Dict[str, Dict],
                             voice_index: Dict[str, Dict], facts_captions: Dict[str, Dict], clip_id: str) -> bool:

    existing_facts_dict = {f['fact_id']: f for f in person_obj.get('facts', [])}
    old_fact_ids: Set[str] = set(existing_facts_dict.keys())

    person_id = person_obj["person_id"]
    face_id = person_obj["face_id"]
    voice_ids = person_obj.get("voice_ids", [])

    print(f"    📊 [DEBUG] Starting fact collection for {person_id}")
    print(f"    📊 [DEBUG] Face ID: {face_id}, Voice IDs: {voice_ids}")
    print(f"    📊 [DEBUG] Existing facts: {len(old_fact_ids)}, Total facts available: {len(facts_captions)}")
    print(f"    📊 [DEBUG] Available fact IDs in facts_captions: {list(facts_captions.keys())}")

    if face_id in face_index:
        face_data = face_index[face_id]
        print(f"    📊 [DEBUG] Found face_id in face_index")
        print(f"    📊 [DEBUG] face_data structure: {face_data.keys()}")

        faces_list = face_data.get("faces", [])
        print(f"    📊 [DEBUG] Number of face entries: {len(faces_list)}")

        face_facts_count = 0
        for idx, face_info in enumerate(faces_list):
            print(f"    📊 [DEBUG] Face entry {idx} keys: {face_info.keys()}")
            #print(f"    📊 [DEBUG] Face entry {idx} content: {face_info}")

            if "fact_id" in face_info:
                fact_id = face_info.get("fact_id", "")
                clip_name = face_info.get("clip_name", "")

                print(f"    📊 [DEBUG] Processing face fact: {fact_id} from clip {clip_name}")
                print(f"    📊 [DEBUG] Is '{fact_id}' in facts_captions? {fact_id in facts_captions}")

                if fact_id not in existing_facts_dict:
                    fact_data = facts_captions.get(fact_id, {})
                    existing_facts_dict[fact_id] = {
                        "fact_id": fact_id,
                        "clip_id": clip_name,
                        "character_level_facts": fact_data.get("character_level_facts", ""),
                        "character_details": fact_data.get("character_details", {}),
                        "source": "face"
                    }
                    face_facts_count += 1
                    print(f"      ✅ Added face fact: {fact_id}")
                else:
                    print(f"      ℹ️ Face fact {fact_id} already exists")
            else:
                print(f"    📊 [DEBUG] Face entry {idx} has NO 'fact_id' key!")

        print(f"    📊 [DEBUG] Added {face_facts_count} facts from face")
    else:
        print(f"    ⚠️ Face ID {face_id} NOT in face_index! Available: {list(face_index.keys())}")

    voice_facts_count = 0
    for voice_id in voice_ids:
        if voice_id in voice_index:
            voice_data = voice_index[voice_id]
            if "fact_id" in voice_data:
                clip_name= voice_data.get("clip_name", "")
                fact_id = voice_data["fact_id"]

                print(f"    📊 [DEBUG] Processing voice fact: {fact_id} from {voice_id}")

                if fact_id not in existing_facts_dict:
                    fact_data = facts_captions.get(fact_id, {})
                    existing_facts_dict[fact_id] = {
                        "fact_id": fact_id,
                        "clip_id": clip_name,
                        "character_level_facts": fact_data.get("character_level_facts", ""),
                        "character_details": fact_data.get("character_details", {}),
                        "source": "voice"
                    }
                    voice_facts_count += 1
                    print(f"      ✅ Added voice fact: {fact_id}")
                else:
                    print(f"      ℹ️ Voice fact {fact_id} already exists")
        else:
            print(f"    📊 [DEBUG] Voice {voice_id} not in voice_index")

    print(f"    📊 [DEBUG] Added {voice_facts_count} facts from voice")

    mentioned_facts_count = 0
    print(f"    📊 [DEBUG] Starting mention check for {person_id}")
    for fact_id, fact_data in facts_captions.items():
        if fact_id in existing_facts_dict:
            print(f"    📊 [DEBUG] Fact {fact_id} already in existing_facts_dict, skip mention")
            continue

        character_level_facts = fact_data.get("character_level_facts", "")
        print(f"    📊 [DEBUG] Checking if '<{person_id}>' in fact {fact_id}")
        print(f"    📊 [DEBUG] Content: {character_level_facts[:100]}")

        if f"<{person_id}>" in character_level_facts:
            existing_facts_dict[fact_id] = {
                "fact_id": fact_id,
                "clip_id": clip_id,
                "character_level_facts": fact_data.get("character_level_facts", ""),
                "character_details": fact_data.get("character_details", {}),
                "source": "mentioned"
            }
            mentioned_facts_count += 1
            print(f"    🔗 {person_id} mentioned in {fact_id}, ADDED")
        else:
            print(f"    📊 [DEBUG] {person_id} NOT mentioned in {fact_id}")

    print(f"    📊 [DEBUG] Added {mentioned_facts_count} facts from mention")

    person_obj['facts'] = list(existing_facts_dict.values())

    all_fact_ids = set(existing_facts_dict.keys())
    newly_added = all_fact_ids - old_fact_ids

    has_changes = len(newly_added) > 0
    if has_changes:
        print(f"    🆕 Added {len(newly_added)} new facts for {person_obj['person_id']}: {len(old_fact_ids)} -> {len(all_fact_ids)}")
        print(f"    📊 [DEBUG] Newly added fact IDs: {newly_added}")
    else:
        print(f"    ℹ️ No new facts for {person_obj['person_id']} (total: {len(all_fact_ids)})")

    return has_changes

def save_person_file(person_obj: Dict, character_dir: str) -> None:
    person_id = person_obj["person_id"]
    output_path = os.path.join(character_dir, f"{person_id}.json")
    clean_person_obj = {
        "person_id": person_obj["person_id"],
        "face_id": person_obj["face_id"],
        "voice_ids": person_obj["voice_ids"],
        "facts": [{
            "fact_id": fact["fact_id"],
            "clip_id": fact["clip_id"],
            "character_level_facts": fact["character_level_facts"],
        }
        for fact in person_obj["facts"]
        ],
        "character_profile": person_obj["character_profile"],
        "name": person_obj.get("name", "")
    }
    save_json(clean_person_obj, output_path)
    print(f"  💾 Saved {person_id} to {output_path}")


def generate_profile_for_person(person_obj: Dict, clip_id: str) -> None:

    inputs = []
    character_facts = person_obj.get("facts", [])

    past_profile = person_obj.get("character_profile", {"description": ""})
    if past_profile and past_profile.get("description", "") not in [""]:
        past_profile_text = f"Past Profile:\n{past_profile['description']}\n\n"
        print("Founded past profile:", past_profile_text)
    else:
        past_profile_text = ""

    fact_details_pairs = []

    for fact_entry in character_facts:

        char_level_fact = fact_entry.get("character_level_facts", "")
        if not char_level_fact:
            continue

        char_details = fact_entry.get("character_details", {})

        pair_text = f"Character-level Fact: {char_level_fact}\n"

        if char_details:
            pair_text += "Details:\n"
            for entity_id, details in char_details.items():
                detail_parts = []
                if "appearance" in details and details["appearance"]:
                    detail_parts.append(f"Appearance: {details['appearance']}")
                if "actions" in details and details["actions"]:
                    detail_parts.append(f"Actions: {details['actions']}")
                if "relation" in details and details["relation"]:
                    detail_parts.append(f"Relation: {details['relation']}")
                if "speech" in details and details["speech"]:
                    detail_parts.append(f"Speech: {details['speech']}")
                if "role" in details and details["role"]:
                    detail_parts.append(f"Role: {details['role']}")

                if detail_parts:
                    pair_text += f"  {entity_id}: {'; '.join(detail_parts)}\n"

        fact_details_pairs.append(pair_text)

    if fact_details_pairs:
        facts_section = "\n".join(fact_details_pairs)
    else:
        facts_section = "No facts available."

    full_context = f"Clip {clip_id} Information:\n\n"
    if past_profile_text:
        full_context += past_profile_text
    full_context += facts_section

    inputs = [{"type": "text", "content": full_context}]
    try:
        response = call_model(inputs, CHARACTER_PROFILE_SUMMARIZATION)
        person_obj["character_profile"] = smart_json_loads(response)
        print(f"    ✅ Generated profile for {person_obj['person_id']}")
    except Exception as e:
        print(f"    ❌ Failed to generate profile for {person_obj['person_id']}: {e}")
        person_obj["character_profile"] = past_profile

def replace_featureID_in_captions(facts_caption: Dict[str, Dict], id_mapping: Dict[str, str]) -> None:

    for fact_id, fact_content in facts_caption.items():
        fact_content["character_level_summary"] = replace_all_ids(
            fact_content.get("character_level_summary", ""), id_mapping
        )

        fact_content["character_level_facts"] = replace_all_ids(
            fact_content.get("character_level_facts", ""), id_mapping
        )

        print(f"  🔄 Replaced IDs in fact {fact_id}")

def create_character_level_facts(facts_path: str, facts_captions: Dict[str, Dict],
                                 clip_id: str, facts_with_featureID_path: str) -> Dict[str, Dict]:

    facts = load_facts(facts_path)
    clip_facts = facts.get(str(clip_id), {})

    if facts_captions:
        first_fact = next(iter(facts_captions.values()))
        clip_facts["character_level_summary"] = first_fact.get("character_level_summary", "")

    facts_list = clip_facts.get("facts", [])

    for fact in facts_list:
        fact_id = fact.get("id")

        if not fact_id:
            continue

        if fact_id in facts_captions:
            caption_data = facts_captions[fact_id]

            fact["character_level_facts"] = caption_data.get("character_level_facts", "")

            print(f"    ✅ Added character-level info to {fact_id}")
        else:
            print(f"    ⚠️ No caption data found for {fact_id}")

    output_data = {str(clip_id): clip_facts}
    save_json(output_data, facts_with_featureID_path)
    print(f"  💾 Saved facts with person IDs to {facts_with_featureID_path}")

    return output_data

def load_voice_index(voices_clip_dir: str) -> Dict[str, Dict]:
    voice_index: Dict[str, Dict] = {}
    if os.path.exists(voices_clip_dir):
        for vf in glob.glob(os.path.join(voices_clip_dir, "*.json")):
            data = load_facts(vf)
            vid = data.get("voice_id")
            if vid:
                voice_index[vid] = data
    print(f"  🔊 Loaded {len(voice_index)} voices")
    return voice_index

def get_unique_person_names(facts_captions: Dict[str, Dict], id_mapping: Dict[str, str]) -> Dict[str, str]:

    person_names = {}
    for fact_id, fact_content in facts_captions.items():
        text = fact_content.get("character_level_facts", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        matches = re.findall(r'<(person_\d+)>\s*\(([^)]+)\)', text)
        for pid, name in matches:
            if pid not in person_names and name.strip():
                person_names[pid] = name.strip()
    return person_names

def process_clip(clip_id: str, faces_dir: str, voices_dir: str,
                equivalence_path: str, character_dir: str,
                caption_file_path: str, snapshot_dir: str, facts_path: str, facts_with_featureID_path, db=None) -> None:

    print(f"\n🎬 Processing clip: {clip_id}")

    face_index = load_all_faces(faces_dir)
    facts_captions = load_facts_captions(caption_file_path, clip_id)
    id_mapping = {}

    if not face_index:
        print("  ⚠️ No faces found, skipping person processing.")
    else:
        existing_persons = load_existing_persons(character_dir)
        voice_index = load_voice_index(voices_dir)
        equivalence_links = load_equivalence_links(equivalence_path, clip_id)
        id_mapping = build_id_mapping(face_index, voice_index, equivalence_links)

    replace_featureID_in_captions(facts_captions, id_mapping)
    print("  🔄 Replaced feature IDs in captions")
    create_character_level_facts(facts_path, facts_captions, clip_id, facts_with_featureID_path)
    print("  💾 Saved facts with character-level feature IDs")

    if not face_index:
        if db:
            db.create_snapshot(str(clip_id))
            print(f"  📸 Snapshot created for no-face clip {clip_id}")
        return

    person_names = get_unique_person_names(facts_captions, id_mapping)

    for face_id, face_data in face_index.items():

        person_obj = create_or_update_person(face_id, existing_persons)
        print(f"  👤 Created/Updated person object for face ID: {face_id}")

        name = person_names.get(person_obj["person_id"])
        if name:
            person_obj["name"] = name

        voices_updated = attach_voices_to_person(person_obj, equivalence_links, voice_index)
        print(f"  🔊 Attached voices to person {person_obj['person_id']}: {voices_updated}")

        facts_updated = collect_facts_for_person(person_obj, face_index, voice_index, facts_captions, clip_id)
        print(f"  📝 Collected facts for person {person_obj['person_id']}: {facts_updated}")

        if voices_updated or facts_updated:
            generate_profile_for_person(person_obj, clip_id)
            print(f"  🖼️ Generated profile for person {person_obj['person_id']}")

        save_person_file(person_obj, character_dir)
        print(f"  💾 Saved person file for {person_obj['person_id']}")
        clips = [f["clip_id"] for f in person_obj['facts']]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, person_obj["person_id"]))
        rag_entry = {
            "person_id": person_obj["person_id"],
            "character_profile": person_obj.get("character_profile", ""),
            "clips": clips,
            "facts": [{
                "fact_id": f["fact_id"],
                "clip_id": f["clip_id"],
                "character_level_facts": f["character_level_facts"],

            }for f in person_obj.get("facts", [])
            ]
        }
        if db:
            db.insert_person(rag_entry, point_id)
            print(f"  🗄️ Inserted/Updated {person_obj['person_id']} in database")

    if db:
        db.create_snapshot(str(clip_id))
        print(f"  📸 Created snapshot for clip {clip_id} at {snapshot_dir}")

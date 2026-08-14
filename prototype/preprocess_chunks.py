import os
from prototype.tools.prompts import gist_generation_template, entity_extraction_template
from tqdm import tqdm
import numpy as np
import json
import argparse
from prototype.tools.api_client import APIClient
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_single_video(filename, file_path, client, dataset):
    print(f"Processing {filename}...")

    with open(os.path.join(file_path, filename), 'r') as f:
        data = json.load(f)

    all_clip_dict = {}
    facts_embeddings = []
    clip_summary_embeddings = []
    fact_counter = 0

    for _, key in enumerate(tqdm(data, desc=f"Clip in {filename}")):
        clip_summary = data[key]['clip_summary']
        clip_summary_embedding = client.obtain_embedding(clip_summary)

        clip_summary_embeddings.append(np.array(clip_summary_embedding, dtype=np.float32))

        character_level_clip_summary = data[key].get('character_level_summary', '')
        scene_clip_summary = data[key].get('scene_description', '')

        all_clip_dict[key] = {'clip_summary': clip_summary,'character_level_clip_summary': character_level_clip_summary, 'scene_clip_summary': scene_clip_summary, 'facts': []}
        facts = data[key].get('facts', [])
        for fact in facts:
            # raw fact text (without text)
            raw_fact_text = fact['description']
            raw_fact_embedding = client.obtain_embedding(raw_fact_text)

            raw_fact_embedding = np.array(raw_fact_embedding, dtype=np.float32)
            # character facts
            character_level_facts = fact.get('character_level_facts', '')
            # scene description
            scene_fact_description = fact.get('scene_description', '')
            # conversation asr
            conversation_asr = fact.get('asr_periods', [])
            # fact uuid
            fact_uuid = fact.get('id', '')
            # fact dict
            fact_dict = {
                "fact_id": fact_counter,
                "raw_fact_text": raw_fact_text,
                "character_level_facts": character_level_facts,
                "asr_periods": conversation_asr,
                "scene_description": scene_fact_description,
                # FIXME: This should be a single image path, currently we have an array
                "image_path": fact['key_frames'][0]['b64_path'] if len(fact['key_frames']) > 0 else "",
                "fact_uuid": fact_uuid,
                "clip_id": key,
                "timestamp": fact['timestamp']
            }
            all_clip_dict[key]['facts'].append(fact_dict)
            facts_embeddings.append(raw_fact_embedding)
            fact_counter += 1

    file_title = os.path.splitext(filename)[0]

    video_embeddings_dict = {'clip_summary': np.stack(clip_summary_embeddings, axis=0),
                             'facts': np.stack(facts_embeddings, axis=0)}
    os.makedirs(f'./processed_data/{dataset}/fact_embeddings', exist_ok=True)
    np.save(f'./processed_data/{dataset}/fact_embeddings/{file_title}_embeddings.npy', video_embeddings_dict)

    os.makedirs(f'./processed_data/{dataset}/fact_metadata', exist_ok=True)
    with open(f'./processed_data/{dataset}/fact_metadata/{file_title}_metadata.json', "w") as f:
        json.dump(all_clip_dict, f, indent=4)
    print(f"Finished {filename}: {fact_counter} facts, \
          fact embeddings shape {video_embeddings_dict['facts'].shape}, \
          clip summary embeddings shape {video_embeddings_dict['clip_summary'].shape}")
def main():
    parser = argparse.ArgumentParser(description="Clip-level embedding extraction")
    parser.add_argument("--dataset", type=str, default='videomme-test', help="Dataset name")
    parser.add_argument("--facts_dir", type=str, default='./processed_data/videomme-test/facts/', help="Facts directory")
    parser.add_argument("--api_key_dir", type=str, default="./config/openai_key.txt", help="Path to OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    parser.add_argument("--max_workers", type=int, default=1, help="Max parallel workers")

    args = parser.parse_args()

    client = APIClient("openai", args.api_key_dir, args.model, args.embedding_model)

    files = [f for f in os.listdir(args.facts_dir) if f.endswith('.json')]
    max_workers = max(1, min(args.max_workers, len(files)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_video, filename, args.facts_dir, client, args.dataset): filename for filename in files}
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

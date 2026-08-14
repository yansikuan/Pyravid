# PyraVid: Hierarchical Multimodal Memory for Long-Horizon Video Reasoning

PyraVid is a hierarchical multimodal memory framework for online long-video
understanding and structure-guided question answering.

[[Paper](https://arxiv.org/abs/2605.17065)]
[[HTML](https://arxiv.org/html/2605.17065)]

> This is an early-stage research release. Dataset files, model weights,
> generated memories, and experiment outputs are not distributed with the
> repository.

## Overview

PyraVid processes a long video as a sequence of short clips and organizes the
resulting observations into a coarse-to-fine memory:

- **Fact memory** stores temporally localized multimodal observations and
  evidence.
- **Clip memory** summarizes events within a local time range.
- **Global memory** maintains a high-level representation of the full video.
- **Character memory** optionally aligns face and voice identities and stores
  character profiles in a local Qdrant database.

During question answering, PyraVid retrieves seed facts, expands through the
memory graph, prunes irrelevant evidence, and iterates until it can produce an
answer.

~~~text
video -> clips -> facts/keyframes -> preprocessing -> hierarchical memory
      -> graph retrieval/expansion/pruning -> answer
~~~

## Repository layout

~~~text
prototype/
  video_extraction/          Video fact and keyframe extraction
  face_voice_processing/     Face, voice, alignment, and character processing
  tasks/                     Multiple-choice and open-question reasoning
  tools/                     API clients, prompts, retrieval, and vector store
  character_processing_online.py
  preprocess_chunks.py
  constructivist_memory.py

scripts/
  init_models/               Local vLLM server launchers
  memory_construction/       Extraction, preprocessing, and graph construction
  question_answering/        Multiple-choice and open-question launchers
  video_segment.sh           Split source videos into 30-second clips

requirements.txt             Main application environment
requirements_omni.txt        Optional Qwen-Omni extraction dependencies
requirements_vllm.txt        Separate vLLM serving environment
pyproject.toml               Package metadata
~~~

## Requirements

- Linux
- Python 3.10--3.12
- <code>ffmpeg</code> and <code>ffprobe</code> on <code>PATH</code>
- NVIDIA GPUs and a compatible CUDA runtime for GPU workflows

The pinned application requirements target Python 3.12 on Linux. Voice
processing uses TorchAudio 2.8 APIs, while vLLM uses Torch 2.9, so keep the
application and serving dependencies in separate environments.

## Installation

Create the main PyraVid environment:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Install the optional Qwen-Omni extraction overlay when needed:

~~~bash
python -m pip install -r requirements_omni.txt --no-build-isolation
~~~

Create a separate environment for model serving:

~~~bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_vllm.txt
~~~

Select Torch wheels compatible with the CUDA runtime available on the target
machine.

## Configuration

Create the local environment file:

~~~bash
cp .env.example .env
~~~

<code>.env.example</code> contains variable names only. Fill
<code>.env</code> with credentials and machine-specific settings;
<code>.env</code> is ignored by Git. Prefer repository-relative paths:

~~~bash
PYRAVID_GRAPH_DIR=./memory/graphs
PYRAVID_MEMORY_EMBEDDING_DIR=./memory/embeddings
~~~

Legacy key files are supported through
<code>PYRAVID_OPENAI_KEY_PATH</code> and <code>PYRAVID_KEY_PATH</code>.
Provider API keys may also be supplied directly through the process
environment.

The local OpenAI-compatible endpoints are configured with
<code>PYRAVID_ANSWER_BASE_URL</code>,
<code>PYRAVID_SELECTION_BASE_URL</code>, and
<code>PYRAVID_LINK_BASE_URL</code>.

## Local model services

The vLLM environment serves three models:

~~~text
scripts/init_models/init_answer_model.sh       answer model, port 8000
scripts/init_models/init_link_model.sh         link model, port 8001
scripts/init_models/init_selection_model.sh    selection model, port 8003
~~~

Run each launcher in a separate terminal:

~~~bash
bash scripts/init_models/init_answer_model.sh
bash scripts/init_models/init_link_model.sh
bash scripts/init_models/init_selection_model.sh
~~~

Before running on another machine, review the CUDA device selection,
tensor-parallel size, maximum model length, and model download directory in
each launcher.

## Data preparation

PyraVid supports Video-MME, LVBench, M3-Bench-web, and M3-Bench-robot
workflows. Obtain each dataset from its official source and comply with its
license.

The current Video-MME test workflow expects:

~~~text
data/
  video_lists/
    videomme_test.txt
  videomme/
    test/                    Numbered 30-second MP4 clips
    test_facts/              One fact JSON file per video
    questions/               One question JSON file per video
    keyframes/               Extracted visual evidence
~~~

Each line in a video-list file points to one directory containing numbered
clips such as <code>0.mp4</code>, <code>1.mp4</code>, and
<code>2.mp4</code>. Facts, questions, graphs, embeddings, and character
snapshots must use the same video ID.

### Segment source videos

Set <code>video_dir</code> and <code>output_dir</code> near the top of
<code>scripts/video_segment.sh</code>, then run:

~~~bash
bash scripts/video_segment.sh
~~~

The script uses FFmpeg to generate 30-second H.264/AAC clips.

### Extract facts and keyframes

Configure <code>.env</code>, or pass a dataset alias and video list
explicitly:

~~~bash
bash scripts/memory_construction/memory_extraction.sh
~~~

~~~bash
bash scripts/memory_construction/memory_extraction.sh \
  videomme data/video_lists/videomme_test.txt
~~~

Supported aliases are <code>videomme</code>, <code>lvbench</code>,
<code>m3web</code>, and <code>m3robot</code>. This step loads Qwen-Omni
locally and therefore requires the Omni dependencies and sufficient GPU
memory.

## Character processing

Character processing is optional for multiple-choice QA and required by the
current open-question workflow when character profiles are queried.

Place the SpeakerLab ERes2NetV2 checkpoint at:

~~~text
models/pretrained_eres2netv2.ckpt
~~~

Configure <code>PYRAVID_CHARACTER_VIDEO_FOLDER</code>,
<code>PYRAVID_CHARACTER_FACTS_PATH</code>, and
<code>PYRAVID_CHARACTER_WORK_DIR</code>, or pass the paths directly:

~~~bash
bash scripts/memory_construction/character_processing_online.sh \
  data/videomme/test \
  data/videomme/test_facts/test.json \
  artifacts/character_processing/videomme
~~~

Character snapshots use this layout:

~~~text
<work_dir>/snapshot/<video_id>/clip_<clip_id>_snapshot/
~~~

## Build hierarchical memory

Preprocess the extracted facts:

~~~bash
bash scripts/memory_construction/preprocess.sh
~~~

Then construct the fact--clip--global graph and its embeddings:

~~~bash
bash scripts/memory_construction/memory_graph_construction.sh
~~~

Both commands load defaults from <code>.env</code>. A dataset can also be
supplied explicitly, for example:

~~~bash
bash scripts/memory_construction/preprocess.sh lvbench
~~~

Generated files are organized under:

~~~text
processed_data/<dataset>/
memory/graphs/<dataset>/<video_id>/
memory/embeddings/<dataset>/<video_id>/
artifacts/
~~~

Use <code>python -m prototype.constructivist_memory --help</code> for the
complete construction and ablation options.

## Question answering

Start the answer and selection services before launching inference.

### Multiple-choice QA

~~~bash
bash scripts/question_answering/run_quesiton_answering.sh
~~~

This launcher runs
<code>prototype/tasks/question_answering_agentic_expand.py</code> with
multimodal answering, graph expansion, pruning, top-level summary context, and
evidence saving.

### Open-question QA

~~~bash
bash scripts/question_answering/run_open_question_answering.sh
~~~

This launcher runs
<code>prototype/tasks/open_question_answering_agentic_expand.py</code> and
uses repository-relative question, graph, embedding, fact, character snapshot,
log, and output paths.

Results and latency reports are written under
<code>artifacts/outputs/</code>.

## Validation

Run checks that do not require datasets, model services, or API credentials:

~~~bash
python -m compileall -q prototype
bash -n scripts/memory_construction/memory_extraction.sh
bash -n scripts/memory_construction/preprocess.sh
bash -n scripts/memory_construction/memory_graph_construction.sh
bash -n scripts/question_answering/run_quesiton_answering.sh
bash -n scripts/question_answering/run_open_question_answering.sh
~~~

GPU-, model-, dataset-, and hosted-API-dependent commands are integration
workflows and require the corresponding external resources.

## Generated files

Datasets, generated memories, local databases, model weights, logs, and
experiment outputs are intentionally excluded from version control. See
<code>.gitignore</code> for the complete list.

## Citation

If you use this repository, please cite:

```bibtex
@article{yan2026pyravid,
  title   = {PyraVid: Hierarchical Multimodal Memory for Long-Horizon Video Reasoning},
  author  = {Yan, Sikuan and Dong, Sicheng and Wang, Haotong and Nie, Ercong and
             Liu, Yilun and Bi, Jinhe and Xu, Yingjie and Schwarzmann, Susanna and
             Trivisonno, Riccardo and Tresp, Volker and Ma, Yunpu},
  journal = {arXiv preprint arXiv:2605.17065},
  year    = {2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Do not commit credentials, dataset
copies, model weights, logs, or generated experiment outputs.

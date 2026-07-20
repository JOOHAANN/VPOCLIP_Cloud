# VPOCLIP – action recognition server

This framework of the demo can detect the unseen human action by tri-model fusion CLIP framework, which source from the unpublished work from youhan. And in this assignment I combined this unseen HAR framework with the server-client system to realize adding a new action while it runs by describing it in plain words: the description goes to an LLM which turns it into a CLIP prompt, and the server registers it on the fly. No retraining.

This framework can be run in: 
1. Webcam real-time detection
2. Curl offline validation by the images
All of the above methods can show the unseen action recognition ability.

## Run it

```bash
cd VPOCLIP_Cloud
pip install -r requirements.txt
cp .env.example .env      # put your OpenAI key in .env
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

First run? You need the model weights too — see [Weights & model files](#weights--model-files) below. Without them the server still boots, but only in mock mode.

Full list of endpoints and a test page: http://127.0.0.1:8000/docs

Webcam client (in another terminal):

```bash
python client/webcam_client.py --server http://127.0.0.1:8000
```

Keys in the window: `a` add an action, `r` get a report, `q` quit.

## Weights & model files

The model weights and the large assets are **not** in this repo (they are too
big for GitHub), so a fresh clone won't do real inference until you add them.
Download the archive from the cloud drive and unzip it inside `VPOCLIP_Cloud/`
so the files land under `local_models/` and `work_dir/`.

> 📦 **Download:** _<paste the cloud-drive link here>_

Layout after unzipping:

```
VPOCLIP_Cloud/
├── local_models/
│   ├── pose_landmarker_full.task          # MediaPipe pose  (~9 MB)
│   ├── yolov5m.pt                         # YOLOv5m detector (~41 MB)
│   └── clip/
│       └── ViT-B-32.pt                    # CLIP encoder     (~338 MB)
└── work_dir/
    └── clipgcn_contrastive_50_5/
        └── run_20260616_210139/
            └── best_model_old.pth         # trained CLIPGCN head (~387 MB)
```

If the weights (or their imports) are missing the server still starts, just in
**mock mode**: `/recognize` returns random predictions so you can still try the
API. `GET /health` will show `mock_mode: true` in that case.

The X3D and CTR-GCN backbone checkpoints are not part of this folder — they live
in their own repos, at the paths referenced in `server/pipeline.py` and
`test_raw_end_to_end.py`.

## Endpoints

- `POST /recognize` : send JPEG frames, get the top action + top-5
- `POST /add_action` : `{name, casual_description}`, adds a new action through the LLM
- `GET /actions` : list the current actions (and which ones are unseen)
- `DELETE /actions/{name}` : remove one
- `POST /report` : short summary of the recent activity
- `GET /health` : status, device, latency

## curl examples

```bash
B=http://127.0.0.1:8000

curl $B/health

# add an action, the description can be in any language
curl -X POST $B/add_action \
  -H "Content-Type: application/json" \
  -d '{"name":"asking for help","casual_description":"The old person asking for help by waving his hands fastly with anxiety"}'

curl $B/actions

# recognize a clip (13 frames extracted from a video)
curl -s -X POST $B/recognize $(for i in $(seq 0 12); do echo -n "-F files=@A57_frame$i.jpg "; done) | python3 -m json.tool

```

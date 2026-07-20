# VPOCLIP – action recognition server

This framework of the demo can detect the unseen human action by tri-model fusion CLIP framework, which source from the unpublished work from youhan. And in this assignment I combined this unseen HAR framework with the server-client system to realize adding a new action while it runs by describing it in plain words: the description goes to an LLM which turns it into a CLIP prompt, and the server registers it on the fly. No retraining.

## Run it

```bash
cd VPOCLIP_Cloud
pip install -r requirements.txt
cp .env.example .env      # put your OpenAI key in .env
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Full list of endpoints and a test page: http://127.0.0.1:8000/docs

Webcam client (in another terminal):

```bash
python client/webcam_client.py --server http://127.0.0.1:8000
```

Keys in the window: `a` add an action, `r` get a report, `q` quit.

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

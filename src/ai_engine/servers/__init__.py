"""Long-lived servers: a resident model, request state, a queue.

Everything here binds a socket, keeps a model loaded for the process
lifetime, and arbitrates between concurrent requests. If a module does not
have all three properties, it does not belong in this folder -- see
workers/ (load once, run a batch, exit) and pipelines/ (one shot, no
socket).

    llm_server.py           text generation, OpenAI-shaped HTTP API,
                            continuous batching (was serve.py -- renamed
                            because every other module in this project is
                            `*_server.py`; the launcher scripts follow the
                            same rule: <module>.sh, e.g. llm_server.sh,
                            image_gen_server.sh, lipsync_server.sh)
    embedding_server.py     embeddings from a QV-band-compressed model
    reranker_server.py      reranking from a QV-band-compressed model
    image_gen_server.py     image generation + the bundled web panel
    lipsync_server.py       Sonic lip-sync video generation (still face +
                            speech audio -> talking clip)
    video_gen_server.py     Wan2.2 text-to-video / image-to-video, serves
                            each rendered mp4 at /videos/<name>.mp4
    image_ui.py             the panel's HTML (presentation only, no model)
    video_ui.py             the video panel's HTML (presentation only)

None of these implement inference. They own a socket, a request/response
shape, and the decision about which backend to load; everything between
device resolution and the forward pass lives in runtime/ and backends/.
"""

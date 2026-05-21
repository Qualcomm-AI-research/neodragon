FROM nvidia/cuda:12.0.0-base-ubuntu22.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        git \
        build-essential \
	curl \
	ffmpeg \
	less \
	python-is-python3 \
	python3 \
	python3-dev \
        sudo \
	vim \
        wget \
        && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3

COPY docker/requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt

WORKDIR /workspace/neodragon

COPY --chmod=755 . /workspace/neodragon/

RUN python3 -m pip install --no-cache-dir --no-dependencies /workspace/neodragon

CMD ["/bin/bash", "--login"]

FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda10.2

ENV TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0 8.6'
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

# Install MMDetection
RUN pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /tmp/mmdetection \
    && cd /tmp/mmdetection \
    && pip install -r requirements/build.txt \
    && pip install --no-cache-dir -e .

RUN jupyter nbextension enable --py widgetsnbextension

FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda10.2

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN jupyter nbextension enable --py widgetsnbextension

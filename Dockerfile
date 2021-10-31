FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda10.2

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
#RUN pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html


RUN jupyter nbextension enable --py widgetsnbextension

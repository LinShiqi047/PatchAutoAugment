# ------------------------------------------------------------------
# python        3.8    (apt)
# torch         1.6.0
# other python add-on
# ==================================================================
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer="Linshiqi <linsq047@mail.ustc.edu.cn>"
RUN  sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt -y update && \
    apt -y upgrade && \
    apt install -y vim && \
    pip --no-cache-dir install -i https://mirrors.ustc.edu.cn/pypi/web/simple pip -U && \
    pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple && \
    pip --no-cache-dir install torch scikit-image \ 
     scikit-video torchvision tqdm msgpack termcolor yacs ipdb opencv-python numpy scipy Pillow \ 
     imgaug tensorboard moviepy ipython matplotlib kornia torchsnooper&& \
    apt autoremove -y && apt autoclean -y

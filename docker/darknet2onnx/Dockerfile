ARG BASE_IMAGE=python:3.7
FROM $BASE_IMAGE
# ARGUMENTS
LABEL maintainer phongphu <phongphu>

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git \
    && rm -rf /var/lib/apt/lists/*

# add pip packages when necessary
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

RUN cd /workspace/ && git clone --branch=main https://github.com/phongphuhanam/pytorch-YOLOv4.git

ARG USER_NAME=dev 
ENV USERNAME $USER_NAME

# ENV HOME /home/$USERNAME
# RUN useradd -m $USERNAME && \
#        echo "$USERNAME:$USERNAME" | chpasswd && \
#        usermod --shell /bin/bash $USERNAME && \
#        usermod -aG sudo $USERNAME && \
#        mkdir -p /etc/sudoers.d && \
#        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
#        chmod 0440 /etc/sudoers.d/$USERNAME && \                                                      
#        # Replace 1000 with your user/group id
#        usermod  --uid 1000 $USERNAME && \
#        groupmod --gid 1000 $USERNAME
   
# RUN ARCH="$(dpkg --print-architecture)" && \
#        curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.5/fixuid-0.5-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - && \
#        chown root:root /usr/local/bin/fixuid && \
#        chmod 4755 /usr/local/bin/fixuid && \
#        mkdir -p /etc/fixuid && \
#        printf "user: $USERNAME\ngroup: $USERNAME\n" > /etc/fixuid/config.yml

# USER root
# RUN echo "${USERNAME}:${USERNAME}" | chpasswd

# USER ${USERNAME}
# WORKDIR /home/${USERNAME}
# ENTRYPOINT ["fixuid"]
ENTRYPOINT ["python", "/workspace/pytorch-YOLOv4/demo_darknet2onnx.py"]

# Our base image
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy the requirements.txt file to our Docker image
ADD requirements.txt .

# Install the requirements.txt
RUN pip install -r requirements.txt

# Install pytorch
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

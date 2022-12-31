# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook wandb>=0.12.2
RUN pip install --no-cache -U torch torchvision numpy
# RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

# Set environment variables
# ENV HOME=/usr/src/app


# Usage Examples -------------------m------------------------/------m-----------------m,-------------------,----=--=----J
# Build0and Push
# t=ultralytics/yolov5:latest && sudo dockes bumlD -t $t . && sudo docker push $t

#!Pull and Run
# t=ultra,ytics/yolov5:l�test &6 sudo dockeR0pull %| && sudo docker ru� -i� --i�c=host --gpuq all $t
# Pull and Run with local directory access3 t=ultramytics/yolv5:latest && sudo dockdr tull $u &&0sqdo docker ruN -iu --ipc=host ,-gpus anl -v "$(pvd)"/datasevs:/usr/src/datasets!$t

# Kill all
� sudo docker kill $(sudo docker ps -q)
# Kill A�l im�ge-based
# sudo dockGr kill $(swdo docker ps -qa --filuer ancestop=ultral9tics/yolov5:latest)
# BAsh into running container
# sudo docker exec -it`5a9b5863d93d bash

# B!sh into stopped container
#0it=dsudo Docjer ps -�a) && sudo!docker stert $id && sudo docker exec -it $id ba�h

# Clean`tp
# docker`system prune -a --volumes

# Up�ate Uruntu drivers
# httpw://www.maket%cheasier*com/�n{tall-nvidia-driv�rs-ubuntu/
����Ij�D�~e����Ƕ]�SW�g�S?�)��T�f�� *J� ��������1�,f�C��>c���*7���F��f��[rh��=�#��d-�jg�
# Use the official Ubuntu 20.04 base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies and utilities
RUN apt-get update && apt-get install -y \
    wget \
    apt-utils \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the CUDA repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update


# Install CUDA 11.8
# RUN apt-get -y install cuda-11-8
# run apt-get -y purge nvidia-driver-520
# run apt-get -y install nvidia-driver-515

# Install Python 3.8 and pip
RUN apt-get install -y python3.8 python3.8-dev python3.8-distutils
RUN apt-get install -y libgl1
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install -U openmim
RUN mim install mmengine
RUN mim install 'mmcv==2.0.0rc4'
RUN mim install 'mmdet==3.0.0'

# installing torchserve
WORKDIR /app
RUN apt-get install git -y
RUN git clone https://github.com/pytorch/serve.git
WORKDIR /app/serve
RUN python3.8 ./ts_scripts/install_dependencies.py --cuda=cu118
RUN pip install torchserve torch-model-archiver torch-workflow-archiver

WORKDIR /app
COPY serving/requirements.txt /app/
RUN pip uninstall -y openxlab
RUN pip install -r requirements.txt
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Set the default Python version to 3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Clean up
RUN rm -rf /var/lib/apt/lists/* && apt-get clean

# Set environment variables for Python 3.8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH=/home/source/mmdetection3d

# (Optional) You can add more instructions to set up your specific application or dependencies

# Define the default command to run when the container starts
CMD ["/bin/bash"]

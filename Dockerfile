FROM    ubuntu:20.04
#nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    libffi-dev \
    build-essential \
    rsync \
    curl \
    wget \
    vim

SHELL ["/bin/bash","--login","-c"]
# Install Miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
        chmod +x ~/miniconda.sh && \
        ~/miniconda.sh -b -p $CONDA_DIR &&\
        rm ~/miniconda.sh
#RUN chmod 755 Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
#RUN ./Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b
#RUN source /root/miniconda3/etc/profile.d/conda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash

#Install Omero
# /root/miniconda3/bin/conda
RUN conda create -y -n cellpose -c ome python=3.8 zeroc-ice36-python omero-py pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly  -c nvidia
# install commands need to be positioned between create and activate to show up in active container at runtime

# moved here to conda activate to install in the following packages to the virtual environment
#adding conda activate to profile to keep it active when bash --login
RUN echo "conda activate cellpose" >> ~/.profile

#set Shell to use new environment

# this needs to be called before the run conda install and run python -m pip install commands
SHELL ["/bin/bash", "--login", "-c"]

## I would prefer the login method, what do you think? seems more robust
#SHELL ["/root/miniconda3/bin/conda", "run", "--no-capture-output","-n", "cellpose", "/bin/bash", "-c"]





RUN python -m pip install cellpose
RUN python -m pip install cellpose --upgrade
RUN conda install scikit-image zarr dask matplotlib statsmodels -c conda-forge
RUN pip install shapely
RUN python -m pip install -i  https://test.pypi.org/simple/ JonasTools


#RUN echo "conda activate cellpose" >> ~/.profile




COPY . .




#ENTRYPOINT can be used instead of CMD in case the parameters given with docker run should not override CMD 
#CMD ["/root/miniconda3/bin/conda", "run","--no-capture-output", "-n", "cellpose", "main_cellpose.py", "ome", "253891", "/PATHMODIFY", "False", "1"]
#CMD["/root/miniconda3/bin/conda","run","--no-capture-output","-n","cellpose"]

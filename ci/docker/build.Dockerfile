FROM continuumio/miniconda3:4.10.3-alpine as yeoda-build
MAINTAINER Florian Roth <florian.roth@geo.tuwien.ac.at>

# define settings
WORKDIR /opt

# create conda environment
ADD docker/yeoda-env.yml /tmp/yeoda-env.yml
RUN conda install -c conda-forge mamba --yes
RUN mamba env create -f /tmp/yeoda-env.yml -p /opt/conda/envs/yeoda -q
RUN bash -c "source activate yeoda && pip install yeoda==0.3.0 -q && pip uninstall opencv-python -yq && pip install opencv-python-headless"

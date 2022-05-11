FROM continuumio/miniconda3:4.10.3-alpine as yeoda-doc
COPY --from=yeoda-build:latest /opt/conda/envs /opt/conda/envs

ENV PATH /opt/conda/envs/yeoda/bin:$PATH
# set environment variables for gdal
ENV PROJ_LIB /opt/conda/envs/yeoda/share/proj
ENV GDAL_DATA /opt/conda/envs/yeoda/share/gdal

ADD docker/.build/notebooks /opt/notebooks
ADD docker/.build/data /data
CMD /bin/bash -c "/opt/conda/envs/yeoda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
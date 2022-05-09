FROM continuumio/miniconda3:4.10.3-alpine as yeoda
COPY --from=yeoda-build:latest /opt/conda/envs /opt/conda/envs

ENV PATH /opt/conda/envs/yeoda/bin:$PATH
# set environment variables for gdal
ENV PROJ_LIB /opt/conda/envs/yeoda/share/proj
ENV GDAL_DATA /opt/conda/envs/yeoda/share/gdal

ADD docker/entry.sh /bin/entry.sh
ENTRYPOINT ["/bin/entry.sh"]
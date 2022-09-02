FROM continuumio/miniconda3:4.10.3-alpine as yeoda-build
MAINTAINER Florian Roth <florian.roth@geo.tuwien.ac.at>

# define settings
WORKDIR /opt

# create conda environment
RUN conda install -c conda-forge mamba --yes
RUN mamba create -p /opt/conda/envs/yeoda -c conda-forge python=3.8 -q
RUN bash -c "source activate yeoda && mamba install -c conda-forge gdal geopandas cartopy jupyter xarray netcdf4 rioxarray dask pyproj && pip install yeoda"

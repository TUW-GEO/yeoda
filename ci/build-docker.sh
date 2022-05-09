#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
pushd "$SCRIPT_DIR" || exit
docker build -t yeoda-build -f docker/build.Dockerfile .
docker build -t tuwgeomrs/yeoda -f docker/yeoda.Dockerfile .
if [ -d /data/SCATSAR_SWI ] && [ -d /data/Sentinel-1_CSAR ]; then
  mkdir -p docker/.build/data
  rsync -auz ../docs/notebooks docker/.build/
  rsync -auz /data/SCATSAR_SWI docker/.build/data/
  rsync -auz /data/Sentinel-1_CSAR docker/.build/data/
  docker build -t tuwgeomrs/yeoda-doc -f docker/yeoda-doc.Dockerfile .
else
  echo "the /data/SCATSAR_SWI and /data/Sentinel directories containing doc test data are missing on this machine"
  echo "skipping building yeoda-doc image."
fi
popd
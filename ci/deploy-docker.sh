#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
pushd "$SCRIPT_DIR" || exit

tag=$1
echo "deploying tag: ${tag}"

docker push tuwgeomrs/yeoda:${tag}
if [ -d /data/SCATSAR_SWI ] && [ -d /data/Sentinel-1_CSAR ]; then
  docker push tuwgeomrs/yeoda-doc:${tag}
else
  echo "the /data/SCATSAR_SWI and /data/Sentinel directories containing doc test data are missing on this machine"
  echo "skipping deploying yeoda-doc image."
fi

popd
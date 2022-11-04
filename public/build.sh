#!/bin/sh
set -e

LOCALREPO=rsandagon
IMAGE=tindertwombly
OLDIMAGE=`docker inspect $LOCALREPO/$IMAGE | grep \"Id\": | awk -F: '{ print $3 }' | sed 's/[", ]//g'`

docker rmi -f $OLDIMAGE || echo 'old image does not exist..'
docker build . -t $LOCALREPO/$IMAGE:latest -f Dockerfile

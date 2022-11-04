#!/bin/bash
set -e

LOCALREPO=rsandagon
IMAGE=tindertwombly-api
DOCKER_REPO=docker.io/rsandagon

VERSION="1.0.0"

# docker login --username $1 --password $2 $DOCKER_REPO
docker tag $LOCALREPO/$IMAGE:latest $DOCKER_REPO/$IMAGE:v$VERSION
docker tag $LOCALREPO/$IMAGE:latest $DOCKER_REPO/$IMAGE:latest
docker push $DOCKER_REPO/$IMAGE:v$VERSION
docker push $DOCKER_REPO/$IMAGE:latest

OLDIMAGE=`docker inspect $LOCALREPO/$IMAGE | grep \"Id\": | awk -F: '{ print $3 }' | sed 's/[", ]//g'`
docker rmi -f $OLDIMAGE || echo 'old image does not exist..'
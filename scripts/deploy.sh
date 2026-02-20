#!/bin/bash
set -e

AWS_REGION=$1
AWS_ACCOUNT_ID=$2
REPOSITORY=$3
IMAGE_TAG=$4

REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
FULL_IMAGE="$REGISTRY/$REPOSITORY:$IMAGE_TAG"

# Authenticate Docker with ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

# Pull the new image
docker pull $FULL_IMAGE

# Stop and remove the old container if it exists
docker stop app || true
docker rm app || true

# Start the new container
docker run -d \
  --name app \
  --restart unless-stopped \
  -p 8080:8080 \
  $FULL_IMAGE

# Verify it started
sleep 5
docker ps | grep app
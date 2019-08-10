SHELL:=/bin/bash


.ONESHELL:
.DEFAULT=all
.PHONY: help test

export PUBLIC_PORT := 8888

export VERSION											:= 3
export PROJECT_NAME								  := siim-acr
export FUNCTION_NAME								:= pub 
export NETWORK_NAME                 := fastai
export DOCKER_REPO									:= musedivision
export WORK_DIR											:=/home/ubuntu/
export uname												:= $(shell uname)
export Proc													:= CPU

# detect if running on AWS GPU
ifeq "$(uname)" "Linux"
	runtime := --runtime=nvidia
	Proc	:= GPU
endif

help: ## This help.
	@echo "_________________________________________________"
	@echo "XXX-     ${Proc} ${runtime}       ${uname}   -XXX"
	@echo "_________________________________________________"
	@echo "CURRENT VERSION: ${VERSION}"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	ecs-cli configure --cluster forsite --region ap-southeast-2 --default-launch-type FARGATE

############################################################
# RUN
############################################################


up: build run

venv:
	source ${PROJECT_NAME}/bin/activate

compose: ## compose
	# launching docker containers
	docker-compose up -d

open: ## open
	open http://localhost:8888

build: ## Build the container
	docker build -t $(PROJECT_NAME)-${FUNCTION_NAME} .

build-nc: ## Build the container without caching
	docker build --no-cache -t $(PROJECT_NAME)-${FUNCTION_NAME} .

restart: ## restart container
	${MAKE} stop start-local

run:
	docker run -d --rm \
					-p 8889:8888 \
					-e LANG=C.UTF-8 \
					-e LC_ALL=C.UTF-8 \
					-e JUPYTER_ENABLE_LAB=no \
					-v $$HOME/data:${WORK_DIR}/data \
					-v $$HOME/code:${WORK_DIR}/code \
					-v $$HOME/Library/Application\ Support/Anki2/atest/collection.media/:${WORK_DIR}/anki \
					-v $$HOME/data/fastai:/root/.fastai \
					-v $$HOME/data/torch:/root/.torch \
					--ipc=host \
					--shm-size 50G \
					$(runtime)\
					--name="$(PROJECT_NAME)-${FUNCTION_NAME}" \
					${cont} \
					jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.password='${JUPYTER_PASSWORD_SHA}'

start-local: ## run local build
	${MAKE} run cont=$(PROJECT_NAME)-${FUNCTION_NAME} 

start: ## start dockerhub container
	${MAKE} run cont=$(DOCKER_REPO)/$(PROJECT_NAME) 

stop: ## stop
	docker stop $(PROJECT_NAME)-${FUNCTION_NAME} $(DOCKER_REPO)/$(PROJECT_NAME) || true

clean_images: ## clean_images
	docker rmi -f $(PROJECT_NAME)-${FUNCTION_NAME} $(DOCKER_REPO)/$(PROJECT_NAME) || true

bash: ## bash
	docker exec -it $(PROJECT_NAME)-${FUNCTION_NAME} /bin/bash

#######################################################################################################################
#   DEPLOYMENT   -- maybe i could upload to docker hub    
#
#  Currently consists of automated docker build on push to master
#  takes so long to install pip dependencies
#                                                                                                   #
########################################################################################################################


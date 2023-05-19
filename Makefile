.PHONY: up exec build start down run ls check vs

# Convenience `make` recipes for Docker Compose.
# See URL below for documentation on Docker Compose.
# https://docs.docker.com/engine/reference/commandline/compose

# **Change `SERVICE` to specify other services and projects.**
SERVICE = tta
COMMAND = /bin/zsh

# `PROJECT` is equivalent to `COMPOSE_PROJECT_NAME`.
# Project names are made unique for each user to prevent name clashes.
PROJECT = "${SERVICE}-${USR}"
PROJECT_ROOT = /opt/tta

# Creates a `.env` file in PWD if it does not exist.
# This will help prevent UID/GID bugs in `docker-compose.yaml`,
# which unfortunately cannot use shell outputs in the file.
# Image names have the usernames appended to them to prevent
# name collisions between different users.
ENV_FILE = .env
GID = $(shell id -g)
UID = $(shell id -u)
GRP = $(shell id -gn)
USR = $(shell id -un)
IMAGE_NAME = "${USR}"

# Makefiles require `$\` at the end of a line for multi-line string values.
# https://www.gnu.org/software/make/manual/html_node/Splitting-Lines.html
ENV_TEXT = "$\
GID=${GID}\n$\
UID=${UID}\n$\
GRP=${GRP}\n$\
USR=${USR}\n$\
IMAGE_NAME=${IMAGE_NAME}\n$\
PROJECT_ROOT=${PROJECT_ROOT}\n$\
"
${ENV_FILE}:  # Creates the `.env` file if it does not exist.
	printf ${ENV_TEXT} >> ${ENV_FILE}

env: ${ENV_FILE}

check:  # Checks if the `.env` file exists.
	@if [ ! -f "${ENV_FILE}" ]; then \
		printf "File \`${ENV_FILE}\` does not exist. " && \
		printf "Run \`make env\` to create \`${ENV_FILE}\`.\n" && \
		exit 1; \
	fi

vs:  # Preempts `.vscode-server` directory ownership issues.
	@mkdir -p ${HOME}/.vscode-server

OVERRIDE_FILE = docker-compose.override.yaml
# Makefiles do not read the initial spaces, hence the inclusion of
# indentation at the end of the line. Not pretty but it works.
# The user's $HOME directory should not be mounted to the container's
# $HOME directory as this will override container configurations.
# Do not mount /home:/home as this will cause the same issue.
# Mount NFS as a bind mount for better performance.
OVERRIDE_BASE = "$\
services:$\
\n  ${SERVICE}:$\
\n    volumes:$\
\n      - ${HOME}:/mnt/home$\
\n$\
\n      - type: bind$\
\n        source: /team/team_blu3/lung$\
\n        target: /lung$\
\n$\
\n      - type: bind$\
\n        source: /team/team_blu3/etc/personal$\
\n        target: /personal$\
\n"
# Create override file for Docker Compose configurations for each user.
# For example, different users may use different host volume directories.
${OVERRIDE_FILE}:
	printf ${OVERRIDE_BASE} >> ${OVERRIDE_FILE}

over: ${OVERRIDE_FILE}

build: check vs  # Start service. Rebuilds the image from the Dockerfile before creating a new container.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up --build -d ${SERVICE}
up: check vs  # Start service. Creates a new container from the image.
	COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} up -d ${SERVICE}
exec:  # Execute service. Enter interactive shell.
	DOCKER_BUILDKIT=1 docker compose -p ${PROJECT} exec ${SERVICE} ${COMMAND}
start:  # Start a stopped service without recreating the container. Useful if the previous container must not deleted.
	docker compose -p ${PROJECT} start ${SERVICE}
down:  # Shut down service and delete containers, volumes, networks, etc.
	docker compose -p ${PROJECT} down
run: check vs  # Used for debugging in cases where service will not start.
	docker compose -p ${PROJECT} run ${SERVICE} ${COMMAND}
ls:  # List all services.
	docker compose ls -a

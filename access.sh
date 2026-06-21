#!/bin/bash
xhost +

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

CCACHE_DIR="${HOME}/.ccache"
mkdir -p "$CCACHE_DIR"
ROBOT_VERSION=52
DIR_HASH=$(echo "$PARENT_DIR" | md5sum | cut -c1-8)
echo "Directory $PARENT_DIR hash: $DIR_HASH"
CONTAINER_NAME="kuavo_ws"
# 使用挑战杯封装镜像
IMAGE_NAME="kuavo_challenge_cup_2026:latest"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo -e "\033[31mError: Docker image '${IMAGE_NAME}' not found.\033[0m"
    echo "Load it first:"
    echo "  docker load -i kuavo_challenge_cup_2026_latest.tar"
    exit 1
fi

show_container_info() {
    local div_line="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n$div_line"
    echo -e "📌 \033[34mContainer Info\033[0m: $CONTAINER_NAME"
    echo -e "📂 \033[32mWorking Directory\033[0m:"
    echo -e "   $PARENT_DIR"
    echo -e "🔗 \033[33mMounted Volumes\033[0m:"
    # docker inspect -f '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' $CONTAINER_NAME

    docker inspect -f '{{range .Mounts}}   {{.Source}} → {{.Destination}}{{println}}{{end}}' $CONTAINER_NAME
    echo -e "$div_line\n"
}

USER_ID=$(id -u)

# 检查是否存在同名容器
EXISTING_CONTAINER=$(docker ps -aq -f name=^/${CONTAINER_NAME}$)

if [[ -n "$EXISTING_CONTAINER" ]]; then
    echo "Container '${CONTAINER_NAME}' already exists."
    
    # 检查容器是否正在运行
        CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)
        
        if [[ "$CONTAINER_STATUS" == "exited" ]] || [[ "$CONTAINER_STATUS" == "created" ]]; then
            echo "Starting container '$CONTAINER_NAME' ..."
            docker start $CONTAINER_NAME
            # 等待容器完全启动
            sleep 1
        elif [[ "$CONTAINER_STATUS" == "running" ]]; then
            echo "Container '$CONTAINER_NAME' is already running."
        else
            echo -e "\033[31mError: Container status is '$CONTAINER_STATUS'. Please check container manually.\033[0m"
            exit 1
        fi
        
        # 自动修改容器内的 ROBOT_VERSION 环境变量
        echo "Updating ROBOT_VERSION to 52 in container..."
        docker exec $CONTAINER_NAME bash -c "
            if grep -q '^export ROBOT_VERSION=' /root/.zshrc; then
                sed -i 's/^export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
            elif grep -q 'export ROBOT_VERSION=' /root/.zshrc; then
                sed -i 's/export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
            else
                echo '' >> /root/.zshrc
                echo '# Auto-configured by docker script' >> /root/.zshrc
                echo 'export ROBOT_VERSION=52' >> /root/.zshrc
            fi
        "
        
        show_container_info
        echo "Exec into container '$CONTAINER_NAME' ..."
        docker exec -it $CONTAINER_NAME zsh
        exit 0
fi

# 创建新容器
if [[ -z "$EXISTING_CONTAINER" ]]; then
    echo "Creating a new container '${CONTAINER_NAME}' based on image '${IMAGE_NAME}' ..."
	docker run -it --net host --gpus all \
		--runtime nvidia \
        --name $CONTAINER_NAME \
		--privileged \
		-v /dev:/dev \
		-v "${HOME}/.ros:/root/.ros" \
		-v "$CCACHE_DIR:/root/.ccache" \
		-v "$PARENT_DIR:/root/kuavo_ws" \
		-v "${HOME}/.config/lejuconfig:/root/.config/lejuconfig" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
        -e CARB_GRAPHICS_API=vulkan \
        -e GDK_SYNCHRONIZE=1 \
		-e HOME=/root \
		--group-add=dialout \
		--ulimit rtprio=99 \
		--cap-add=sys_nice \
		--ipc=host \
		-e DISPLAY=$DISPLAY \
		-e ROBOT_VERSION=52 \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		${IMAGE_NAME} \
		bash -c "
			# 修改 .zshrc 中的 ROBOT_VERSION
			if grep -q '^export ROBOT_VERSION=' /root/.zshrc; then
				sed -i 's/^export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
			elif grep -q 'export ROBOT_VERSION=' /root/.zshrc; then
				sed -i 's/export ROBOT_VERSION=.*/export ROBOT_VERSION=52/' /root/.zshrc
			else
				echo '' >> /root/.zshrc
				echo '# Auto-configured by docker script' >> /root/.zshrc
				echo 'export ROBOT_VERSION=52' >> /root/.zshrc
			fi
			exec zsh
		"
fi

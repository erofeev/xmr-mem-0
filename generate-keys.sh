#\!/bin/bash

# Генерация API ключей для проектов
# Формат: {project}_{user}_{hash}
# Пользователи: user00-user20 (21 пользователь на проект)

PROJECTS=("sport" "terra" "datashowcase" "trialprj")
NGINX_DIR="/opt/mcp-team-memory/nginx"

for PROJECT in "${PROJECTS[@]}"; do
    CONF_FILE="$NGINX_DIR/api-keys-${PROJECT}.conf"
    echo "# API Keys for Project ${PROJECT^^} (21 users)" > "$CONF_FILE"
    echo "# Generated: $(date)" >> "$CONF_FILE"
    echo "" >> "$CONF_FILE"
    
    for i in $(seq -w 0 20); do
        USER="user${i}"
        # Генерируем хеш: project + user + random
        HASH=$(echo -n "${PROJECT}_${USER}_$(openssl rand -hex 16)" | sha256sum | cut -d" " -f1 | head -c 32)
        KEY="${PROJECT}_${USER}_${HASH}"
        echo "\"${KEY}\" 1;  # ${PROJECT}_${USER}" >> "$CONF_FILE"
    done
    
    echo "Created: $CONF_FILE"
done

echo ""
echo "API Keys files generated in $NGINX_DIR"

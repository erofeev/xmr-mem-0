# MCP Team Memory Server

Self-hosted Memory + Knowledge Base для команд разработки.

## Проекты

- **Terra** (7 разработчиков): 
- **Sport** (5 разработчиков): 

## Технологии

- **Mem0 MCP Server** — память и Knowledge Graph
- **PostgreSQL + pgvector** — векторный поиск
- **Neo4j** — Knowledge Graph (связи между знаниями)
- **Ollama** — локальные LLM и embeddings (без OpenAI!)
- **Nginx** — reverse proxy с API key аутентификацией

## Быстрый старт

```bash
# Создать .env с паролями
cp .env.example .env

# Запустить
docker compose up -d

# Проверить
curl http://SERVER_IP/health
```

## Конфигурация клиента

Файл `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "project-memory": {
      "url": "http://SERVER_IP/PROJECT/mcp/sse",
      "transport": "sse",
      "headers": {
        "X-API-Key": "YOUR_API_KEY"
      }
    }
  }
}
```

## Стоимость

~2000 ₽/мес (Timeweb Cloud120, 12GB RAM)

# MCP Team Memory Server v3.1

Self-hosted платформа долгосрочной памяти для AI-ассистентов (Claude Code, Cursor, GPT) с поддержкой множества проектов и пользователей.

## Что это?

**MCP Team Memory** позволяет AI-ассистентам помнить информацию между сессиями. Вместо того чтобы каждый раз объяснять контекст проекта, предпочтения, решённые проблемы — ассистент запоминает это навсегда.

---

## Архитектура v3.1 — Cipher in Aggregator Mode

### Ключевые изменения в v3.1

| Было (v3.0) | Стало (v3.1) |
|-------------|--------------|
| Unified MCP → Cipher API | **Cipher SSE MCP** напрямую |
| HTTP-to-Cipher proxy | Прямое MCP соединение |
| Ограниченный набор tools | **17 полных tools** из Cipher |
| Отсутствие Graphiti tools | **Graphiti агрегирован** в Cipher |

### Возможности

- **Memory Context** — сохранение и поиск памяти между сессиями
- **Graphiti Timeline** — временной граф знаний с эволюцией
- **ask_cipher** — AI-агент для сложных memory-задач
- **Reasoning Memory** — сохранение цепочек рассуждений

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────────┐
│                   v3.1 Architecture (Aggregator Mode)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Nginx API Gateway                      │   │
│  │         /cipher/mcp/sse  (SSE transport for MCP)         │   │
│  │         /cipher/mcp      (POST endpoint)                  │   │
│  │              API Key authentication via ?key=             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Cipher (Aggregator)                    │   │
│  │              MCP Server with SSE Transport                │   │
│  │                      Port 3000                            │   │
│  │                                                           │   │
│  │  17 Tools:                                                │   │
│  │  • add_memory, search_nodes, search_memory_facts          │   │
│  │  • cipher_memory_search, cipher_store_reasoning_memory    │   │
│  │  • cipher_extract_reasoning_steps, cipher_evaluate_*      │   │
│  │  • ask_cipher (AI agent for complex tasks)                │   │
│  │  • Graphiti: get_episodes, delete_episode, get_status...  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                   │
│  ┌───────────────────────┐     ┌───────────────────────────┐    │
│  │      Graphiti         │     │         Qdrant            │    │
│  │   Temporal Knowledge  │     │    Vector Database        │    │
│  │   (via FalkorDB)      │     │    (Embeddings)           │    │
│  └───────────────────────┘     └───────────────────────────┘    │
│                                                                  │
│  ┌───────────────────────┐     ┌───────────────────────────┐    │
│  │      PostgreSQL       │     │        Ollama             │    │
│  │   (Cipher + Graphiti) │     │  (Local Embeddings)       │    │
│  └───────────────────────┘     └───────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Подключение к Claude Code

### .mcp.json (в корне проекта)

```json
{
  "mcpServers": {
    "team-memory": {
      "type": "sse",
      "url": "http://46.149.66.107/cipher/mcp/sse?key=YOUR_PROJECT_API_KEY"
    }
  }
}
```

### Получение API ключа

Ключи генерируются для каждого проекта. Формат: `{project}_{user}_{hash}`

Примеры существующих ключей:
- `terra_user00_ba0320b3087de665b32ded64f1ca1522`
- `datashowcase_user00_9a2a97e56581f3708dbc14999ae03b48`
- `sport_user00_...`

---

## Доступные инструменты (17 tools)

### Memory Operations
| Tool | Описание |
|------|----------|
| `add_memory` | Добавить новую память |
| `cipher_memory_search` | Поиск по памяти с AI-фильтрацией |
| `cipher_store_reasoning_memory` | Сохранить reasoning chain |
| `cipher_extract_and_operate_memory` | Извлечь и обработать память |

### Graphiti (Temporal Knowledge Graph)
| Tool | Описание |
|------|----------|
| `search_nodes` | Поиск узлов в графе знаний |
| `search_memory_facts` | Поиск фактов (edges) |
| `get_episodes` | Получить эпизоды |
| `delete_episode` | Удалить эпизод |
| `delete_entity_edge` | Удалить связь |
| `get_entity_edge` | Получить связь |
| `get_status` | Статус Graphiti |
| `clear_graph` | Очистить граф (осторожно\!) |

### Reasoning & Analysis
| Tool | Описание |
|------|----------|
| `cipher_extract_reasoning_steps` | Извлечь шаги рассуждений |
| `cipher_evaluate_reasoning` | Оценить качество рассуждений |
| `cipher_search_reasoning_patterns` | Поиск паттернов рассуждений |

### Utilities
| Tool | Описание |
|------|----------|
| `cipher_bash` | Выполнить bash команду |
| `ask_cipher` | AI-агент для сложных memory-задач |

---

## Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/cipher/mcp/sse?key=...` | GET | SSE для MCP подключения |
| `/cipher/mcp?key=...&sessionId=...` | POST | POST для MCP сообщений |
| `/cipher/health` | GET | Health check |
| `/cipher/api/...` | * | REST API (требует key) |

---

## Запуск

```bash
cd /opt/mcp-team-memory
docker compose up -d
```

## Мониторинг

```bash
# Статус контейнеров
docker compose ps

# Логи Cipher
docker logs mcp-cipher -f

# Health check
curl http://46.149.66.107/cipher/health
```

---

## Legacy: Unified MCP Servers

Для обратной совместимости сохранены unified-* серверы:
- `/terra/mcp` → unified-terra
- `/sport/mcp` → unified-sport
- `/datashowcase/mcp` → unified-datashowcase

Рекомендуется переход на `/cipher/mcp/sse` для полного функционала.

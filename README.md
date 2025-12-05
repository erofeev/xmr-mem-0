# MCP Team Memory Server

Self-hosted платформа долгосрочной памяти для AI-ассистентов (Claude Code, Cursor, GPT) с поддержкой множества проектов и пользователей.

## Что это?

**MCP Team Memory** позволяет AI-ассистентам помнить информацию между сессиями. Вместо того чтобы каждый раз объяснять контекст проекта, предпочтения, решённые проблемы — ассистент запоминает это навсегда.

### Три компонента платформы

| Компонент | Что делает | Для чего использовать |
|-----------|------------|----------------------|
| **Memory (Mem0)** | Автоматически извлекает факты из разговора | Личные предпочтения, контекст задач |
| **Knowledge Base** | Хранит документы с семантическим поиском | Документация проекта, архитектура |
| **Graphiti** | Temporal knowledge graph (сущности + связи) | Зависимости, архитектурные связи, история изменений |

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Team Memory Server                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │    Nginx     │──│   Gateway    │  │   Shared Services    │  │
│  │   (port 80)  │  │   (legacy)   │  │                      │  │
│  │              │  └──────────────┘  │  ┌─────────────────┐ │  │
│  │ API Key Auth │                    │  │    Graphiti     │ │  │
│  │ + Routing    │                    │  │  (1 instance)   │ │  │
│  └──────┬───────┘                    │  │  + FalkorDB     │ │  │
│         │                            │  │  + Anthropic LLM│ │  │
│         │                            │  │  group_id=proj  │ │  │
│         ▼                            │  └─────────────────┘ │  │
│  ┌──────────────────────────────┐   │                      │  │
│  │    Per-Project Services       │   │  ┌─────────────────┐ │  │
│  │                               │   │  │   PostgreSQL    │ │  │
│  │  ┌─────────────────────────┐ │   │  │   (pgvector)    │ │  │
│  │  │     unified-{proj}      │ │   │  │   DB per proj   │ │  │
│  │  │ FastMCP SSE Transport   │ │   │  └─────────────────┘ │  │
│  │  │ - mem0_* tools          │ │   │                      │  │
│  │  │ - kb_* tools            │ │   │  ┌─────────────────┐ │  │
│  │  │ - graphiti_* tools      │ │   │  │     Neo4j       │ │  │
│  │  └───────────┬─────────────┘ │   │  │   (for Mem0)    │ │  │
│  │              │               │   │  └─────────────────┘ │  │
│  │              ▼               │   │                      │  │
│  │  ┌─────────────────────────┐ │   │  ┌─────────────────┐ │  │
│  │  │      mem0-{proj}        │ │   │  │     Ollama      │ │  │
│  │  │   Memory extraction     │ │   │  │  Embeddings:    │ │  │
│  │  │   + Semantic search     │ │   │  │  nomic-embed-   │ │  │
│  │  │   + Anthropic Claude    │ │   │  │     text        │ │  │
│  │  └─────────────────────────┘ │   │  └─────────────────┘ │  │
│  │                               │   └──────────────────────┘  │
│  │  Projects: terra, sport,      │                             │
│  │  datashowcase, trialprj       │                             │
│  └──────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые особенности архитектуры

1. **LLM через Anthropic Claude** — Graphiti и Mem0 используют claude-3-haiku для экстракции фактов
2. **Ollama только для embeddings** — nomic-embed-text для семантического поиска
3. **Один Graphiti на все проекты** — изоляция через `group_id`
4. **Отдельные Mem0 на проект** — полная изоляция данных памяти
5. **SSE транспорт** — стабильный протокол для Claude Code

---

## Мультипроектность

### Текущие проекты

| Проект | Endpoint | Пользователи |
|--------|----------|--------------|
| **terra** | `/terra/sse?key=...` | user00-user20 |
| **sport** | `/sport/sse?key=...` | user00-user20 |
| **datashowcase** | `/datashowcase/sse?key=...` | user00-user20 |
| **trialprj** | `/trialprj/sse?key=...` | user00-user20 |

**Всего: 4 проекта × 21 пользователь = 84 API ключа**

### Формат API ключей

```
{project}_{user}_{hash}

Примеры:
- terra_user00_ba0320b3087de665b32ded64f1ca1522
- sport_user05_edfd0d7f0cc3d1d19a2c9f4082b0924e
- datashowcase_user10_9a2a97e56581f3708dbc14999ae03b48
```

### Изоляция данных

```
┌─────────────────────────────────────────────────────────────────┐
│                       Data Isolation                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Memory (Mem0):    Per-project + Per-user                      │
│   ├── terra_alice   ← изолировано                               │
│   ├── terra_bob     ← изолировано                               │
│   └── sport_coach   ← изолировано                               │
│                                                                  │
│   Knowledge Base:   Per-project (shared within project)         │
│   ├── terra KB      ← общая для всех user в terra               │
│   └── sport KB      ← общая для всех user в sport               │
│                                                                  │
│   Graphiti:         Per-project via group_id (shared)           │
│   └── Single instance, group_id = project_name                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Быстрый старт

### Подключение к Claude Code

Создайте `.mcp.json` в корне вашего проекта:

```json
{
  "mcpServers": {
    "team-memory": {
      "type": "sse",
      "url": "http://46.149.66.107/terra/sse?key=terra_user00_ba0320b3087de665b32ded64f1ca1522"
    }
  }
}
```

### Проверка подключения

```bash
# Health check
curl http://46.149.66.107/terra/health
# → {"status":"healthy","project":"terra"}

# SSE endpoint (должен вернуть 200)
curl -I "http://46.149.66.107/terra/sse?key=terra_user00_..." -H "Accept: text/event-stream"
# → HTTP/1.1 200 OK
# → Content-Type: text/event-stream

# Auth check (без ключа = 401)
curl -I "http://46.149.66.107/terra/sse" -H "Accept: text/event-stream"
# → HTTP/1.1 401 Unauthorized
```

---

## Доступные инструменты

### Memory (Mem0) — Долгосрочная память

```python
mem0_add(content, user_id?, metadata?)     # Добавить запись
mem0_search(query, user_id?, limit?)       # Семантический поиск
mem0_get_all(user_id?)                     # Все записи пользователя
```

**Пример:**
```python
# Сохранить контекст
mem0_add(
    content="Разработчик работает над модулем авторизации, использует JWT",
    user_id="dev1"
)

# Найти в новой сессии
mem0_search(query="над чем работает", user_id="dev1")
```

### Knowledge Base — База документации

```python
kb_create(name, documents)    # Создать KB из документов
kb_search(name, query, top_k?) # Поиск по KB
kb_list()                      # Список баз
```

**Пример:**
```python
# Создать базу документации
kb_create(
    name="architecture",
    documents=[# Auth Module\nJWT tokens stored in cookies...]
)

# Искать информацию
kb_search(name="architecture", query="как работает авторизация")
```

### Graphiti — Temporal Knowledge Graph

```python
graphiti_add_episode(content, name?, source?)  # Добавить эпизод
graphiti_search(query, num_results?)           # Поиск в графе
graphiti_get_episodes(last_n?)                 # Последние эпизоды
get_project_info()                             # Информация о проекте
```

**Пример:**
```python
# Сохранить архитектурное решение
graphiti_add_episode(
    content="Решили использовать PostgreSQL RLS для изоляции тенантов",
    name="architecture-decision",
    source="team-meeting"
)

# Найти связанные решения
graphiti_search(query="изоляция данных тенанты")
```

**Особенности Graphiti:**
- Temporal graph — хранит историю изменений
- Изоляция через group_id (= project_name)
- LLM экстракция сущностей через Anthropic Claude
- Embeddings через Ollama (nomic-embed-text)

---

## Инфраструктура

### Контейнеры

| Container | Purpose | Memory Limit |
|-----------|---------|--------------|
| mcp-nginx | API Gateway + Auth | - |
| mcp-gateway | Legacy gateway | 256MB |
| unified-{project} | MCP Server | 512MB per project |
| mem0-{project} | Memory service | 512MB per project |
| graphiti | Knowledge graph | 768MB (shared) |
| mcp-postgres | PostgreSQL + pgvector | 512MB |
| mcp-neo4j | Neo4j for Mem0 | 1GB |
| mcp-falkordb | FalkorDB for Graphiti | 1GB |
| mcp-ollama | Ollama embeddings | 4GB |

### Ресурсы

- **Текущее использование:** ~3GB RAM для 4 проектов
- **Доступно:** 11GB RAM
- **Ёмкость:** ~15-20 проектов комфортно

### Файловая структура

```
/opt/mcp-team-memory/
├── docker-compose.yml       # Конфигурация контейнеров
├── .env                     # API ключи (ANTHROPIC_API_KEY и др.)
├── nginx/
│   ├── nginx.conf           # Маршрутизация и прокси
│   ├── api-keys-terra.conf  # Ключи проекта terra
│   ├── api-keys-sport.conf  # Ключи проекта sport
│   ├── api-keys-datashowcase.conf
│   └── api-keys-trialprj.conf
├── unified-mcp/
│   ├── main.py              # FastMCP сервер
│   └── Dockerfile
├── mem0-src/
│   └── mem0-server/         # Mem0 с Anthropic интеграцией
├── graphiti/
│   └── config.yaml          # Graphiti конфигурация
├── gateway/                  # Legacy gateway
└── postgres/
    └── init.sql             # Инициализация БД
```

---

## Добавление нового проекта

### 1. Генерация API ключей

```bash
# На сервере
./generate-keys.sh  # Редактировать PROJECTS в скрипте
```

### 2. Создание PostgreSQL базы

```bash
docker exec mcp-postgres psql -U mem0 -d postgres -c "CREATE DATABASE mem0_newproject;"
docker exec mcp-postgres psql -U mem0 -d mem0_newproject -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Добавление в docker-compose.yml

```yaml
# Добавить секции:
mem0-newproject:
  build: ./mem0-src/mem0-server
  environment:
    - POSTGRES_DB=mem0_newproject
    - LLM_PROVIDER=anthropic
    # ... остальные переменные как у других проектов

unified-newproject:
  build: ./unified-mcp
  environment:
    - CURRENT_PROJECT_ID=newproject
    - MEM0_API_URL=http://mem0-newproject:8000
    # ... остальные переменные

# Добавить volumes:
volumes:
  newproject-history:
  newproject-kb-data:
  newproject-kb-index:
```

### 4. Обновление nginx.conf

```nginx
map $arg_key $newproject_valid {
    default 0;
    include /etc/nginx/api-keys-newproject.conf;
}

# В server блоке:
location = /newproject/sse {
    if ($newproject_valid = 0) { return 401; }
    proxy_pass http://unified-newproject:8080/sse;
}
# ... остальные locations
```

### 5. Применение изменений

```bash
docker compose up -d --build
docker exec mcp-nginx nginx -s reload
```

---

## Troubleshooting

### SSE возвращает 401

- Проверьте формат ключа: `{project}_{user}_{hash}`
- Убедитесь что ключ есть в `/opt/mcp-team-memory/nginx/api-keys-{project}.conf`
- Перезагрузите nginx: `docker exec mcp-nginx nginx -s reload`

### Health возвращает unhealthy

```bash
# Проверить логи
docker logs unified-{project} --tail 50
docker logs mem0-{project} --tail 50

# Перезапустить
docker compose restart unified-{project} mem0-{project}
```

### Graphiti не работает

```bash
# Проверить health
docker logs graphiti --tail 50

# Graphiti использует Anthropic — проверить ключ
grep ANTHROPIC_API_KEY .env
```

### Mem0 не сохраняет

```bash
# Проверить PostgreSQL
docker exec mcp-postgres psql -U mem0 -d mem0_{project} -c "SELECT count(*) FROM memories;"

# Проверить Neo4j
docker logs mcp-neo4j --tail 20
```

---

## Безопасность

1. **API ключи в URL** — передаются через query parameter `?key=`
2. **Nginx валидация** — map директивы проверяют ключи
3. **Нет HTTPS** — для production используйте reverse proxy с SSL
4. **Изоляция проектов** — данные не пересекаются между проектами

---

## Лицензия

MIT

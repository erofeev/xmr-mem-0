# Custom Graphiti MCP with Anthropic support
FROM zepai/knowledge-graph-mcp:standalone

# Install anthropic SDK into the MCP server's venv using uv
WORKDIR /app/mcp
RUN /root/.local/bin/uv pip install --python /app/mcp/.venv/bin/python 'anthropic>=0.49.0'

# Verify anthropic is available in venv
RUN /app/mcp/.venv/bin/python -c "import anthropic; print('Anthropic SDK installed in venv:', anthropic.__version__)"
RUN /app/mcp/.venv/bin/python -c "from graphiti_core.llm_client.anthropic_client import AnthropicClient; print('AnthropicClient available')"

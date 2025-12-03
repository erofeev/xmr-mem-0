import express from "express";
import fs from "fs";

const app = express();
app.use(express.json());

// Load API keys
const loadApiKeys = (project) => {
  const filePath = `/etc/nginx/api-keys-${project}.conf`;
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    const keys = new Set();
    content.split("\n").forEach(line => {
      const match = line.match(/"([^"]+)"\s+1;/);
      if (match) keys.add(match[1]);
    });
    return keys;
  } catch (e) {
    console.error(`Failed to load keys for ${project}:`, e.message);
    return new Set();
  }
};

const terraKeys = loadApiKeys("terra");
const sportKeys = loadApiKeys("sport");
const sessions = new Map(); // session_id -> project mapping

console.log(`Terra keys loaded: ${terraKeys.size}`);
console.log(`Sport keys loaded: ${sportKeys.size}`);

const validateKey = (key) => {
  if (terraKeys.has(key)) return "terra";
  if (sportKeys.has(key)) return "sport";
  return null;
};

// SSE endpoint - proxy to backend
app.get("/sse/", async (req, res) => {
  const apiKey = req.query.key;
  
  if (!apiKey) {
    return res.status(401).json({ error: "API key required" });
  }

  const project = validateKey(apiKey);
  if (!project) {
    return res.status(401).json({ error: "Invalid API key" });
  }

  console.log(`[SSE] ${project} - new connection`);

  const backendUrl = `http://mcp-${project}:8080/sse/`;
  
  try {
    const response = await fetch(backendUrl, {
      headers: { "Accept": "text/event-stream" }
    });

    if (!response.ok) {
      return res.status(response.status).json({ error: "Backend error" });
    }

    // Set SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    // Stream response, rewriting paths
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const pump = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          let text = decoder.decode(value);
          
          // Rewrite path: /sse/messages/ -> /mcp/sse/messages/
          // Also extract session_id to track which project it belongs to
          const sessionMatch = text.match(/session_id=([a-f0-9]+)/);
          if (sessionMatch) {
            sessions.set(sessionMatch[1], project);
            console.log(`[SSE] Registered session ${sessionMatch[1]} for ${project}`);
          }
          
          text = text.replace(/data: \/sse\//g, "data: /mcp/sse/");
          res.write(text);
        }
      } catch (e) {
        console.error(`[SSE] Stream error: ${e.message}`);
      }
    };

    pump();

    req.on("close", () => {
      console.log(`[SSE] ${project} - connection closed`);
      reader.cancel();
    });

  } catch (err) {
    console.error(`[SSE] Backend error: ${err.message}`);
    res.status(502).json({ error: "Backend connection failed" });
  }
});

// Messages endpoint - proxy POST
app.post("/sse/messages/", async (req, res) => {
  const sessionId = req.query.session_id;
  
  if (!sessionId) {
    return res.status(400).json({ error: "session_id required" });
  }
  
  // Get project from session map
  const project = sessions.get(sessionId);
  
  if (!project) {
    console.log(`[POST] Unknown session ${sessionId}, trying terra`);
    // Fallback to terra
  }

  const targetProject = project || "terra";
  const backendUrl = `http://mcp-${targetProject}:8080/sse/messages/?session_id=${sessionId}`;
  
  console.log(`[POST] ${targetProject} - session ${sessionId}`);

  try {
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body)
    });

    const contentType = response.headers.get("content-type");
    
    if (contentType?.includes("text/event-stream")) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(decoder.decode(value));
      }
      res.end();
    } else {
      const data = await response.text();
      res.setHeader("Content-Type", contentType || "application/json");
      res.status(response.status).send(data);
    }

  } catch (err) {
    console.error(`[POST] Error: ${err.message}`);
    res.status(502).json({ error: "Backend request failed" });
  }
});

// Health
app.get("/health", (req, res) => {
  res.json({ status: "ok", terraKeys: terraKeys.size, sportKeys: sportKeys.size, sessions: sessions.size });
});

app.get("/", (req, res) => {
  res.json({ message: "MCP Gateway", usage: "/sse/?key=API_KEY" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`MCP Gateway running on port ${PORT}`);
});

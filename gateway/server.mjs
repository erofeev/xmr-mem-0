import express from "express";
import EventSource from "eventsource";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";

const app = express();
app.use(express.json());

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
const sessions = new Map();

const validateKey = (key) => {
  if (terraKeys.has(key)) return "terra";
  if (sportKeys.has(key)) return "sport";
  return null;
};

const connectToBackend = async (project) => {
  const backendUrl = `http://mcp-${project}:8080/sse/`;
  
  return new Promise((resolve, reject) => {
    const es = new EventSource(backendUrl);
    
    es.addEventListener("endpoint", (event) => {
      const match = event.data.match(/session_id=([a-f0-9]+)/);
      if (match) {
        resolve({
          sessionId: match[1],
          messagesUrl: `http://mcp-${project}:8080/sse/messages/?session_id=${match[1]}`,
          eventSource: es
        });
      }
    });

    es.onerror = (err) => {
      es.close();
      reject(new Error("Backend connection failed"));
    };

    setTimeout(() => {
      es.close();
      reject(new Error("Backend connection timeout"));
    }, 10000);
  });
};

// GET /sse/?key=API_KEY
app.get("/sse/", async (req, res) => {
  const apiKey = req.query.key || req.headers["x-api-key"];
  
  if (!apiKey) {
    return res.status(401).json({ error: "API key required" });
  }

  const project = validateKey(apiKey);
  if (!project) {
    return res.status(401).json({ error: "Invalid API key" });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    const backend = await connectToBackend(project);
    const clientSessionId = uuidv4().replace(/-/g, "");
    
    sessions.set(clientSessionId, {
      project,
      apiKey,
      backendSessionId: backend.sessionId,
      messagesUrl: backend.messagesUrl,
      eventSource: backend.eventSource
    });

    // Return full path including /mcp/ prefix
    res.write(`event: endpoint\ndata: /mcp/sse/messages/?session_id=${clientSessionId}\n\n`);

    backend.eventSource.addEventListener("message", (event) => {
      res.write(`event: message\ndata: ${event.data}\n\n`);
    });

    req.on("close", () => {
      backend.eventSource.close();
      sessions.delete(clientSessionId);
      console.log(`Session ${clientSessionId} closed`);
    });

  } catch (err) {
    console.error("Backend connection error:", err);
    res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

// POST /sse/messages/?session_id=XXX
app.post("/sse/messages/", async (req, res) => {
  const sessionId = req.query.session_id;
  
  if (!sessionId) {
    return res.status(400).json({ error: "session_id required" });
  }

  const session = sessions.get(sessionId);
  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }

  try {
    const response = await fetch(session.messagesUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body)
    });

    const data = await response.text();
    
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.write(`event: message\ndata: ${data}\n\n`);
    res.end();

  } catch (err) {
    console.error("Backend request error:", err);
    res.status(500).json({ error: "Backend request failed" });
  }
});

app.get("/health", (req, res) => {
  res.json({ 
    status: "ok", 
    sessions: sessions.size,
    terraKeys: terraKeys.size,
    sportKeys: sportKeys.size
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`MCP Gateway running on port ${PORT}`);
  console.log(`Terra keys loaded: ${terraKeys.size}`);
  console.log(`Sport keys loaded: ${sportKeys.size}`);
});

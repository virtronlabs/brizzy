import express from 'express';
import { ChromaClient } from 'chromadb';
import path from 'path';

const app = express();
const port = 3000;

app.use(express.json());

// Function to embed text (using Nomic API)
async function embeddingFunction(texts) {
  try {
    // Add a minimum length check
    if (!texts || texts.trim().length < 2) {
      console.warn('Text too short for embedding');
      return Array(384).fill(0); // Return a zero vector of appropriate length
    }

    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'nomic-embed-text', input: texts }),
    });

    if (!response.ok) {
      throw new Error(`Nomic API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.error('Error embedding text:', error);
    return Array(384).fill(0); // Return a zero vector in case of error
  }
}

// Initialize ChromaDB client
const chroma = new ChromaClient();

// Create or get the collection
let collection;
try {
  collection = await chroma.createCollection({
    name: "virtron_metaverse_v7",
    embeddingFunction: { generate: embeddingFunction },
  });
} catch (error) {
  if (error.name === "ChromaUniqueError") {
    collection = await chroma.getCollection({
      name: "virtron_metaverse_v7",
    });
  } else {
    throw error;
  }
}

// Add documents to the collection with embeddings
const documents = [
  "Virtron Metaverse is a virtual world where users can create, explore, and interact with various environments and experiences. It focuses on user-generated content and community-driven development.",
  "The Virtron Metaverse economy is powered by the Virtron token, which allows users to buy, sell, and trade virtual assets, land, and experiences. It also enables participation in governance and decision-making within the metaverse.",
];

const embeddings = await Promise.all(documents.map(embeddingFunction));

await collection.add({
  ids: ["virtron_doc_1", "virtron_doc_2"],
  embeddings: embeddings,
  documents: documents,
});

// API endpoint for chat
app.post('/api/virtron-chat', async (req, res) => {
  console.log('Received /api/virtron-chat request:', req.body);
  const query = req.body.query;
  if (!query) {
    return res.status(400).json({ error: 'Query is required' });
  }

  try {
    const queryEmbedding = await embeddingFunction(query);
    console.log("queryEmbedding length: ", queryEmbedding.length);

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 5,
    });
    console.log("chroma results: ", results);

    const context = results.documents[0].join('\n');
    const ollamaPrompt = `Context: ${context}\n\nUser: ${query}\n\nVirtron AI:`;
    console.log("ollama prompt: ", ollamaPrompt);

    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        model: 'gemma3:1b', 
        prompt: ollamaPrompt,
        stream: false  // Disable streaming for simpler response handling
      }),
    });

    if (!ollamaResponse.ok) {
      console.error("Ollama API Error:", ollamaResponse.status, ollamaResponse.statusText);
      return res.status(500).json({error: "Ollama API Error"});
    }

    const responseData = await ollamaResponse.json();
    console.log("Ollama response:", responseData);

    res.json({ response: responseData.response });
  } catch (error) {
    console.error('Error in /api/virtron-chat:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Serve static files
app.use(express.static(path.join(process.cwd())));

// Serve the index.html file
app.get('/', (req, res) => {
  res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
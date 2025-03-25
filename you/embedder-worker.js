// embedder-worker.js
self.onmessage = async (event) => {
  const { text } = event.data;
  
  try {
    // Simulate embedding generation 
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'nomic-embed-text', input: text }),
    });

    if (!response.ok) {
      throw new Error(`Embedding API error: ${response.status}`);
    }

    const data = await response.json();
    self.postMessage({ 
      embedding: data.embedding,
      success: true 
    });
  } catch (error) {
    console.error('Embedding worker error:', error);
    self.postMessage({ 
      error: error.message,
      success: false 
    });
  }
};
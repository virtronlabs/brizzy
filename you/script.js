// script.js
class VirtronChatClient {
  constructor() {
    this.embeddingWorker = new Worker('embedder-worker.js');
    this.setupWorkerListeners();
  }

  setupWorkerListeners() {
    this.embeddingWorker.onmessage = (event) => {
      const { embedding, success, error } = event.data;
      if (success) {
        this.processEmbedding(embedding);
      } else {
        console.error('Embedding generation failed:', error);
      }
    };
  }

  generateEmbedding(text) {
    this.embeddingWorker.postMessage({ text });
  }

  async sendChatMessage(query) {
    // Generate embedding in worker
    this.generateEmbedding(query);

    // Simultaneously prepare for chat request
    try {
      const response = await fetch('/api/virtron-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      this.displayResponse(data.response);
    } catch (error) {
      console.error('Chat request failed:', error);
    }
  }

  processEmbedding(embedding) {
    // Optional: You could send embedding to server or use locally
    console.log('Embedding generated:', embedding);
  }

  displayResponse(response) {
    // Update UI with response
    document.getElementById('chat-response').textContent = response;
  }
}

// Initialize client
const virtronChat = new VirtronChatClient();

// Example usage
document.getElementById('chat-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const queryInput = document.getElementById('query-input');
  virtronChat.sendChatMessage(queryInput.value);
});
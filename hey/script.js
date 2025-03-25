const chatLog = document.getElementById('chat-log');
const promptInput = document.getElementById('prompt-input');
const sendButton = document.getElementById('send-button');

sendButton.addEventListener('click', async () => {
  const prompt = promptInput.value;
  if (prompt.trim() === '') return;

  // Display user's message
  chatLog.innerHTML += `<p><strong>User:</strong> ${prompt}</p>`;

  try {
    // Send prompt to your Node.js server (which will use ChromaDB)
    const response = await fetch('/api/virtron-chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: prompt }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    chatLog.innerHTML += `<p><strong>Virtron AI:</strong> ${data.response}</p>`;
  } catch (error) {
    console.error('Error:', error);
    chatLog.innerHTML += `<p><strong>Error:</strong> ${error.message}</p>`;
  }

  promptInput.value = ''; // Clear input
  chatLog.scrollTop = chatLog.scrollHeight; // Scroll to bottom
});
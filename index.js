import { config } from 'dotenv';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';
import multer from 'multer';
import speech from '@google-cloud/speech';
import textToSpeech from '@google-cloud/text-to-speech';
import fs from 'fs';

// Load environment variables from .env
config();

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Retrieve API keys and configuration from environment variables
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = process.env.INDEX_NAME || 'google-embedding-index';
const PORT = process.env.PORT || 3001;

// Initialize Google Cloud clients with error handling
let speechClient;
let ttsClient;

try {
    const credentialsPath = path.join(__dirname, 'config', 'google-credentials.json');
    
    if (!fs.existsSync(credentialsPath)) {
        throw new Error(`Google Cloud credentials not found at ${credentialsPath}`);
    }

    speechClient = new speech.SpeechClient({
        keyFilename: credentialsPath
    });
    
    ttsClient = new textToSpeech.TextToSpeechClient({
        keyFilename: credentialsPath
    });
    
    console.log('Google Cloud clients initialized successfully');
} catch (error) {
    console.error('Error initializing Google Cloud clients:', error);
    console.error('Please ensure you have placed your Google Cloud credentials in the config directory');
}

// Configure multer for handling file uploads
const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    }
});

class HybridConversationMemory {
  constructor(options = {}) {
    this.maxSizeBytes = options.maxSizeBytes || 5 * 1024 * 1024; // 5MB default
    this.maxMessageCount = options.maxMessageCount || 20; // Limit total messages
    this.messages = [];
    this.embeddingModel = options.embeddingModel; // Optional custom embedding model
  }

  // Lightweight size estimation to avoid constant Blob creation
  _estimateMessageSize(message) {
    return JSON.stringify(message).length;
  }

  // Simplified semantic relevance scoring
  async _calculateRelevanceScore(message, currentContext) {
    try {
      // If no embedding model, use basic text-based comparison
      if (!this.embeddingModel) {
        const textScore = this._basicTextSimilarity(message.text, currentContext);
        return textScore;
      }

      // Use embedding-based similarity if model available
      const messageEmbedding = await this.embeddingModel.embedContent(message.text);
      const contextEmbedding = await this.embeddingModel.embedContent(currentContext);
      
      return this._cosineSimilarity(
        messageEmbedding.embedding.values, 
        contextEmbedding.embedding.values
      );
    } catch (error) {
      console.warn('Relevance calculation error:', error);
      return 0;
    }
  }

  // Basic text similarity as fallback
  _basicTextSimilarity(text1, text2) {
    const words1 = new Set(text1.toLowerCase().split(/\W+/));
    const words2 = new Set(text2.toLowerCase().split(/\W+/));
    
    const intersection = [...words1].filter(word => words2.has(word));
    return intersection.length / Math.sqrt(words1.size * words2.size);
  }

  // Simplified cosine similarity (placeholder)
  _cosineSimilarity(vec1, vec2) {
    if (vec1.length !== vec2.length) return 0;
    
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      mag1 += vec1[i] * vec1[i];
      mag2 += vec2[i] * vec2[i];
    }
    
    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2));
  }

  // Intelligent message pruning
  async prune(currentContext) {
    if (this.messages.length <= this.maxMessageCount) return;

    // Calculate relevance scores
    const scoredMessages = await Promise.all(
      this.messages.map(async (message, index) => ({
        index,
        message,
        score: await this._calculateRelevanceScore(message, currentContext)
      }))
    );

    // Sort by lowest relevance
    const sortedByLeastRelevant = scoredMessages
      .sort((a, b) => a.score - b.score);

    // Remove least relevant messages
    const messagesToRemove = sortedByLeastRelevant
      .slice(0, this.messages.length - this.maxMessageCount)
      .map(item => item.index);

    // Remove messages in reverse to maintain index integrity
    messagesToRemove
      .sort((a, b) => b - a)
      .forEach(index => this.messages.splice(index, 1));
  }

  // Add a new message with intelligent management
  async addMessage(message, currentContext) {
    // Estimate size and potentially remove older messages
    const messageSize = this._estimateMessageSize(message);
    
    // If adding this message would exceed size, prune
    if (this.messages.reduce((sum, m) => sum + this._estimateMessageSize(m), 0) + messageSize > this.maxSizeBytes) {
      await this.prune(currentContext);
    }

    // Add new message
    this.messages.push({
      ...message,
      timestamp: Date.now()
    });
  }

  // Get most relevant context
  async getRelevantContext(currentQuery, maxContextSize = 1024 * 1024) {
    if (this.messages.length === 0) return '';

    // Sort messages by relevance to current query
    const scoredMessages = await Promise.all(
      this.messages.map(async (message) => ({
        message,
        score: await this._calculateRelevanceScore(message, currentQuery)
      }))
    );

    // Sort by relevance (highest first)
    const sortedMessages = scoredMessages
      .sort((a, b) => b.score - a.score)
      .map(item => item.message);

    // Collect most relevant messages within size limit
    let contextString = '';
    const contextMessages = [];

    for (let msg of sortedMessages) {
      const msgText = JSON.stringify(msg);
      if (contextString.length + msgText.length <= maxContextSize) {
        contextMessages.push(msg);
        contextString += msgText + '\n---\n';
      } else {
        break;
      }
    }

    return contextString;
  }
}

// Example usage in RAG function
async function queryWithRAG(userQuestion, conversationMemory) {
  try {
    // Get relevant context
    const relevantContext = await conversationMemory.getRelevantContext(userQuestion);

    const prompt = `Conversation Context: ${relevantContext}

Current Question: ${userQuestion}

Answer the question considering the previous context:`;

    const result = await generationModel.generateContent(prompt);
    
    // Add this interaction to conversation memory
    await conversationMemory.addMessage({
      type: 'query',
      text: userQuestion
    }, relevantContext);

    await conversationMemory.addMessage({
      type: 'response',
      text: result.response.text()
    }, userQuestion);

    return result.response.text();
  } catch (error) {
    console.error("Error in RAG query:", error);
    return "Sorry, I encountered an error while processing your question.";
  }
}

// Initialization
const conversationMemory = new HybridConversationMemory({
  maxSizeBytes: 5 * 1024 * 1024,  // 5MB
  maxMessageCount: 20,
  embeddingModel: embeddingModel  // Use the existing embedding model
});

module.exports = { 
  HybridConversationMemory, 
  queryWithRAG,
  conversationMemory 
};

// Initialize Gemini
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
const generationModel = genAI.getGenerativeModel({ 
  model: "gemini-2.0-flash",
  systemInstruction: `You are Virtron Labs' AI assistant, designed to provide expert knowledge in esports, combat sports, and gaming-related topics. 
  Your responses should be direct, knowledgeable, and engaging, reflecting the brand's authoritative and innovative tone. 
  Avoid disclaimers like "I am an AI" and keep answers concise yet informative.`
});

// This is the correct way to get embeddings from Google's API
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" }); 

// Initialize Pinecone
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(INDEX_NAME);

// Function to generate embeddings with Google's embedding model
async function generateEmbedding(text) {
  try {
    const result = await embeddingModel.embedContent(text);
    return result.embedding.values;
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}

// Function to index a document
async function indexDocument(id, text, metadata = {}) {
  try {
    const embedding = await generateEmbedding(text);
    
    await index.upsert([{
      id: id,
      values: embedding,
      metadata: { ...metadata, text }
    }]);
    
    console.log(`Indexed document ${id}`);
  } catch (error) {
    console.error(`Error indexing document ${id}:`, error);
  }
}

// RAG query function
async function queryWithRAG(userQuestion) {
  try {
    const questionEmbedding = await generateEmbedding(userQuestion);

    const queryResult = await index.query({
      vector: questionEmbedding,
      topK: 3,
      includeMetadata: true
    });

    console.log("Query matches:", JSON.stringify(queryResult.matches, null, 2));

    if (queryResult.matches.length === 0) {
      return "No relevant information found.";
    }

    const relevantDocs = queryResult.matches
      .filter(match => match.metadata && match.metadata.text)
      .map(match => match.metadata.text);

    if (relevantDocs.length === 0) {
      return "Found matches but couldn't extract text content.";
    }

    const context = relevantDocs[0];

    const prompt = `Based on this information: "${context}"
    
Question: ${userQuestion}

Answer:`;

    const result = await generationModel.generateContent(prompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Error in RAG query:", error);
    return "Sorry, I encountered an error while processing your question. Please try again.";
  }
}

// Initialize Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// API endpoint for chat
app.post('/api/chat', async (req, res) => {
    try {
        const { question } = req.body;
        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        const answer = await queryWithRAG(question);
        res.json({ answer, confidence: 0.95 });
    } catch (error) {
        console.error('Error processing request:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

// API endpoint for speech-to-text
app.post('/api/speech-to-text', upload.single('audio'), async (req, res) => {
    try {
        if (!speechClient) {
            return res.status(500).json({ error: 'Speech-to-Text service not initialized. Please check your Google Cloud credentials.' });
        }

        if (!req.file) {
            return res.status(400).json({ error: 'No audio file provided' });
        }

        const audio = {
            content: req.file.buffer.toString('base64'),
        };
        const config = {
            encoding: 'WEBM_OPUS',
            sampleRateHertz: 48000,
            languageCode: 'en-US',
        };
        const request = {
            audio: audio,
            config: config,
        };

        const [response] = await speechClient.recognize(request);
        const transcription = response.results
            .map(result => result.alternatives[0].transcript)
            .join('\n');

        res.json({ text: transcription });
    } catch (error) {
        console.error('Error processing speech:', error);
        res.status(500).json({ 
            error: 'Error processing speech',
            details: error.message 
        });
    }
});

// // API endpoint for text-to-speech
// app.post('/api/text-to-speech', async (req, res) => {
//     try {
//         if (!ttsClient) {
//             return res.status(500).json({ error: 'Text-to-Speech service not initialized. Please check your Google Cloud credentials.' });
//         }

//         const { text } = req.body;
//         if (!text) {
//             return res.status(400).json({ error: 'No text provided' });
//         }

//         const request = {
//             input: { text },
//             voice: { languageCode: 'en-US', ssmlGender: 'NEUTRAL' },
//             audioConfig: { audioEncoding: 'MP3' },
//         };

//         const [response] = await ttsClient.synthesizeSpeech(request);
//         const audioContent = response.audioContent;

//         res.set('Content-Type', 'audio/mp3');
//         res.send(audioContent);
//     } catch (error) {
//         console.error('Error generating speech:', error);
//         res.status(500).json({ 
//             error: 'Error generating speech',
//             details: error.message 
//         });
//     }
// });

app.post('/api/text-to-speech', async (req, res) => {
  try {
      if (!ttsClient) {
          return res.status(500).json({ error: 'Text-to-Speech service not initialized. Please check your Google Cloud credentials.' });
      }

      const { text } = req.body;
      if (!text) {
          return res.status(400).json({ error: 'No text provided' });
      }

      const request = {
          input: { text: text },
          voice: {
              languageCode: 'en-US',
              name: 'en-US-Chirp-HD-F',  // Use a Neural2 voice (female)
              ssmlGender: 'FEMALE',
          },
          audioConfig: {
              audioEncoding: 'LINEAR16',
              speakingRate: 1,  // Speed up the speech by 10%
              sampleRateHertz: 16000,  // 16 kHz sample rate
          },
      };

      const [response] = await ttsClient.synthesizeSpeech(request);
      const audioContent = response.audioContent;

      res.set('Content-Type', 'audio/L16; rate=16000');
      res.send(audioContent);
  } catch (error) {
      console.error('Error generating speech:', error);
      res.status(500).json({
          error: 'Error generating speech',
          details: error.message
      });
  }
});


// Serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server starting on port ${PORT}...`);
    console.log(`Frontend available at http://localhost:${PORT}`);
    console.log(`API endpoints available at http://localhost:${PORT}/api/chat`);
});

import { config } from 'dotenv';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';

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
      topK: 1,
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

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

// Helper to initialize Gemini SDK client lazily
function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not defined in the environment. Please configure it in the Secrets panel.");
  }
  return new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Increase body size limit to support photo uploads as Base64 data URLs
  app.use(express.json({ limit: "15mb" }));
  app.use(express.urlencoded({ limit: "15mb", extended: true }));

  // API Route - Health Check
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  // API Route - Analyze Room Photo
  app.post("/api/analyze", async (req, res) => {
    try {
      const { image, mimeType, roomTypeHint } = req.body;

      if (!image) {
        return res.status(400).json({ error: "No image data provided." });
      }

      // Check for Gemini API key first
      let ai;
      try {
        ai = getGeminiClient();
      } catch (keyError: any) {
        return res.status(500).json({
          error: "Gemini API Configuration Missing",
          details: keyError.message,
        });
      }

      // Prepare image parts for Gemini
      // Strip out the data:image/*;base64, prefix if present
      const base64Data = image.includes(",") ? image.split(",")[1] : image;
      const cleanMimeType = mimeType || "image/jpeg";

      const imagePart = {
        inlineData: {
          data: base64Data,
          mimeType: cleanMimeType,
        },
      };

      const userPrompt = `Analyze this room photograph ${roomTypeHint ? `(which is likely a ${roomTypeHint})` : ""}. 
Assess its current organization, identify clutter hotspots, suggest 1-2 small rewarding projects, and provide concrete storage recommendations. 
Be constructive, warm, helpful, and structural in your feedback.`;

      const responseSchema = {
        type: Type.OBJECT,
        properties: {
          roomType: {
            type: Type.STRING,
            description: "Determined type of room (e.g., Bedroom, Living Room, Kitchen, Home Office, Closet, Bathroom, Entryway, etc.)"
          },
          cleanlinessScore: {
            type: Type.INTEGER,
            description: "A score from 1 (extremely cluttered/messy) to 10 (spotless, perfectly organized)."
          },
          clutterLevel: {
            type: Type.STRING,
            description: "Overall clutter status: Low, Medium, or High."
          },
          generalObservations: {
            type: Type.STRING,
            description: "A warm, encouraging, yet professional 2-3 sentence overview of the current space layout, layout qualities, and clutter patterns."
          },
          declutterSteps: {
            type: Type.ARRAY,
            description: "Actionable, clear steps to immediately declutter this room.",
            items: {
              type: Type.OBJECT,
              properties: {
                item: { type: Type.STRING, description: "The specific item or area (e.g., bedside table, laundry pile, cluttered desk, floor garbage)." },
                issue: { type: Type.STRING, description: "The specific organization or clutter issue identified." },
                suggestion: { type: Type.STRING, description: "A clear, actionable, friendly recommendation to resolve it." },
                priority: { type: Type.STRING, description: "Priority level: High, Medium, or Low." }
              },
              required: ["item", "issue", "suggestion", "priority"]
            }
          },
          organizationProjects: {
            type: Type.ARRAY,
            description: "A list of exactly 1 or 2 small-scale, fun, high-reward organization projects appropriate for this room.",
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING, description: "Project title (e.g., 'The 15-Minute Drawer Sort')" },
                description: { type: Type.STRING, description: "Brief, motivating summary of the project." },
                estimatedTime: { type: Type.STRING, description: "Estimated time (e.g., '20 mins', '1 hour')" },
                materialsNeeded: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Helpful household items needed (e.g., '1 cardboard box', 'post-it notes')"
                },
                steps: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "Clear step-by-step walkthrough."
                }
              },
              required: ["title", "description", "estimatedTime", "materialsNeeded", "steps"]
            }
          },
          storageSolutions: {
            type: Type.ARRAY,
            description: "Creative storage ideas or life-hacks specifically suited for this type of room.",
            items: {
              type: Type.OBJECT,
              properties: {
                category: { type: Type.STRING, description: "Category (e.g., Vertical Storage, Clever Containers, Under-bed, Labels)" },
                idea: { type: Type.STRING, description: "The core storage idea or organization hack." },
                benefit: { type: Type.STRING, description: "Why this helps keep the room clean and beautifully organized." }
              },
              required: ["category", "idea", "benefit"]
            }
          }
        },
        required: [
          "roomType",
          "cleanlinessScore",
          "clutterLevel",
          "generalObservations",
          "declutterSteps",
          "organizationProjects",
          "storageSolutions"
        ]
      };

      // Call Gemini 3.1 Pro Preview as mandated by the room image understanding feature requirement
      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: { parts: [imagePart, { text: userPrompt }] },
        config: {
          responseMimeType: "application/json",
          responseSchema: responseSchema,
        },
      });

      const resultText = response.text;
      if (!resultText) {
        throw new Error("No analysis response returned from Gemini.");
      }

      const parsedResult = JSON.parse(resultText.trim());
      res.json(parsedResult);
    } catch (error: any) {
      console.error("Error analyzing image:", error);
      res.status(500).json({
        error: "Failed to analyze room photograph",
        details: error.message || error,
      });
    }
  });

  // API Route - Companion Chatbot
  app.post("/api/chat", async (req, res) => {
    try {
      const { message, history } = req.body;

      if (!message) {
        return res.status(400).json({ error: "Message is required." });
      }

      let ai;
      try {
        ai = getGeminiClient();
      } catch (keyError: any) {
        return res.status(500).json({
          error: "Gemini API Configuration Missing",
          details: keyError.message,
        });
      }

      // Structure system instruction to give the chatbot its specific role (as required)
      const systemInstruction = `You are "Spruce", an encouraging, friendly, and professional home organizer, interior decluttering expert, and space-saving consultant.
Your mission is to help the user clear clutter, organize their living areas, form solid storage habits, and design personalized cleaning routines.
Provide highly practical, structured suggestions with step-by-step items or bullet points. 
Be extremely warm and compassionate — never judge mess or clutter. Help them feel motivated, peaceful, and in control of their space.`;

      // Initialize the multi-turn chat session with historical messages
      // Format history: Array<{ role: 'user' | 'model', parts: [{ text: string }] }>
      const formattedHistory = (history || []).map((h: any) => ({
        role: h.role === "assistant" ? "model" : h.role,
        parts: h.parts || [{ text: h.text }],
      }));

      // Use gemini-3.5-flash for general chat tasks as requested
      const chat = ai.chats.create({
        model: "gemini-3.5-flash",
        config: {
          systemInstruction: systemInstruction,
        },
        history: formattedHistory,
      });

      const response = await chat.sendMessage({ message: message });
      const replyText = response.text;

      res.json({ text: replyText });
    } catch (error: any) {
      console.error("Chat error:", error);
      res.status(500).json({
        error: "Failed to process chat message",
        details: error.message || error,
      });
    }
  });

  // Vite Integration & Static Assets
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();

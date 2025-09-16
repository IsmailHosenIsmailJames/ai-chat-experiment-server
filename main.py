import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part, Content, Part
import google.auth

# --- Configuration ---
# Attempt to get project ID from the environment or authenticated session
try:
    _, PROJECT_ID = google.auth.default()
except (google.auth.exceptions.DefaultCredentialsError, TypeError):
    PROJECT_ID = "poised-defender-469608" # Fallback project ID

LOCATION = "us-central1"  # e.g., "us-central1"

available_models = [
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-embedding-001',
]

MODEL_NAME = available_models[2]

# --- Service Account Authentication ---
# The presence of 'poised-defender-469608-h4-d2407799542a.json' suggests
# you are using a service account. Before running the server, set the
# environment variable in your terminal like this:
#
# export GOOGLE_APPLICATION_CREDENTIALS="poised-defender-469608-h4-d2407799542a.json"
#
# If you run this in a Google Cloud environment (like Cloud Run), auth is often automatic.

# Initialize Vertex AI
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    print("Please ensure you have authenticated correctly and the project ID is valid.")


# --- FastAPI App ---
app = FastAPI(
    title="Gemini Proxy Server",
    description="A simple server to forward prompts to the Google Gemini API via Vertex AI.",
    version="1.0.0",
)

# --- Pydantic Models ---
# This defines the structure of the request body for the /chat endpoint
class ChatHistoryItem(BaseModel):
    role: str
    parts: List[str]

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[ChatHistoryItem]] = None
    system_prompt: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    history: List[ChatHistoryItem]


# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint to check if the server is running."""
    return {"message": "Gemini Proxy Server is running."}


@app.post("/chat", response_model=ChatResponse)
async def get_gemini_response(request: ChatRequest):
    """
    Receives a prompt and chat history, and forwards it to the Google Gemini API.
    """
    try:
        # Construct the system prompt from system_prompt and user_info
        system_instructions = []
        if request.system_prompt:
            system_instructions.append(request.system_prompt)

        if request.user_info:
            user_info_str = "\n".join([f"- {key}: {value}" for key, value in request.user_info.items()])
            system_instructions.append(f"**User Information:**\n{user_info_str}")

        full_system_prompt = "\n\n".join(system_instructions)

        # Load the generative model with the combined system prompt
        if full_system_prompt:
            model = GenerativeModel(MODEL_NAME, system_instruction=[full_system_prompt])
        else:
            model = GenerativeModel(MODEL_NAME)

        # Convert history to Content objects

        history_content = []
        if request.history:
            for item in request.history:
                history_content.append(Content(role=item.role, parts=[Part.from_text(p) for p in item.parts]))

        # Start a chat session
        chat = model.start_chat(history=history_content)

        # Send the prompt to the model
        response = await chat.send_message_async(request.prompt)

        # Extract and return the text response
        if response.candidates:
            # Add the user's prompt and the model's response to the history
            updated_history = [
                {"role": item.role, "parts": [part.text for part in item.parts]}
                for item in chat.history
            ]

            return {
                "response": response.candidates[0].content.parts[0].text,
                "history": updated_history,
            }
        else:
            # Handle cases where the model doesn't return a candidate
            raise HTTPException(status_code=500, detail="No response candidate found from the model.")

    except Exception as e:
        # Handle potential errors from the API call
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

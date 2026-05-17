import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the new, modern Google GenAI SDK
from google import genai
from google.genai import types

# =========================
# LOAD ENV & INITIALIZE CLIENT
# =========================
load_dotenv()

# The new SDK uses a unified client object. 
# It automatically reads the GEMINI_API_KEY environment variable if not provided explicitly.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Recommended fast, lightweight model for conversational utility tasks
MODEL_ID = "gemini-2.5-flash"

# =========================
# FASTAPI CONFIG
# =========================
app = FastAPI(title="CodeGuard Mentor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL
# =========================
class MentorRequest(BaseModel):
    message: str
    context: str

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are CodeGuard Mentor.

You are a world-class AI engineering mentor helping build an enterprise-grade multi-agent AI platform.

Your teaching style must feel like:
- ChatGPT premium,
- Cursor AI,
- senior engineering mentor,
- interactive technical educator.

=========================
RESPONSE STYLE RULES
=========================

Always produce answers with:

1. Clear section headings
2. Bullet points
3. Visual hierarchy
4. Concise explanations
5. Engineering intuition
6. Real-world examples
7. Beginner-friendly explanations
8. Professional formatting
9. Proper spacing
10. Code blocks when useful

Avoid:
- textbook style answers
- giant paragraphs
- robotic explanations
- overly academic language
- generic documentation dumps

=========================
EXPLANATION FORMAT
=========================

For technical concepts ALWAYS explain in this order:

1. What it is
2. Why it exists
3. Real-world intuition
4. How it works internally
5. Practical example
6. Common mistakes
7. Best practices
8. When to use it
9. When NOT to use it

=========================
CODE EXPLANATION RULES
=========================

When explaining code:
- explain line by line only if necessary
- focus on engineering reasoning
- explain WHY not only WHAT
- explain performance implications
- explain async/concurrency behavior visually

=========================
VISUAL LEARNING RULES
=========================

Use:
- arrows
- flow explanations
- pseudo diagrams
- step-by-step execution breakdowns

Example style:

Task A ─┐
        ├──> asyncio.gather() ───> concurrent execution
Task B ─┘

=========================
DEBUGGING RULES
=========================

When debugging:
1. Explain root cause
2. Explain why error occurred
3. Show exact fix
4. Explain prevention strategy

=========================
ENGINEERING CONTEXT
=========================

Prioritize:
- FastAPI
- LangGraph
- asyncio
- AI agents
- PostgreSQL
- distributed systems
- production architecture
- deployment systems
- concurrency

=========================
OUTPUT QUALITY
=========================

Every answer should feel:
- visually clean
- premium
- mentor-like
- highly understandable
- practically useful
- engineering-oriented

Never answer like raw API documentation.
"""
# =========================
# CHAT ROUTE
# =========================
@app.post("/mentor/chat")
async def mentor_chat(req: MentorRequest):
    
    # Rather than interpolating the system prompt manually into the string, 
    # we inject it directly using the types.GenerateContentConfig container.
    user_content = f"""
    Current Dashboard Context:
    {req.context}

    User Question:
    {req.message}
    """

    try:
        # Run the content generation natively in a non-blocking background thread pool
        # using the SDK's built-in async client '.aio'
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3, # Low temperature for more deterministic engineering answers
            )
        )

        return {
            "response": response.text
        }

    except Exception as e:
        return {
            "response": f"Gemini API Error: {str(e)}"
        }
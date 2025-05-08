from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
from typing import Optional
import uvicorn  
import json
import re


app = FastAPI(title="llama2 Document Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TextRequest(BaseModel):
    text: str
    contentType: Optional[str] = "llama2"  # Can be "llama2", "investment", "banking", etc.
    model_size: Optional[str] = "7b"
    advanced_analysis: Optional[bool] = False

class AnalysisResponse(BaseModel):
    summary: str
    roadmap: str
    key_concepts: Optional[list] = None
    difficulty_level: Optional[str] = None
    is_llama2_domain: bool
    domain_confidence: float

def is_llama2_related(text: str) -> tuple[bool, float]:
    """Check if the text is llama2-related and return confidence score."""
    llama2_keywords = [
        'llama2', 'financial', 'money', 'investment', 'stock', 'market', 'banking',
        'accounting', 'revenue', 'profit', 'loss', 'budget', 'cash', 'credit', 'debit',
        'loan', 'interest', 'rate', 'currency', 'exchange', 'trading', 'portfolio',
        'asset', 'liability', 'equity', 'balance', 'sheet', 'income', 'statement',
        'cash flow', 'dividend', 'share', 'bond', 'security', 'derivative', 'option',
        'futures', 'hedge', 'risk', 'return', 'capital', 'fund', 'mutual', 'etf',
        'ipo', 'merger', 'acquisition', 'valuation', 'price', 'cost', 'expense',
        'tax', 'audit', 'compliance', 'regulation', 'compliance', 'forex', 'forecast',
        'analysis', 'ratio', 'margin', 'leverage', 'debt', 'equity', 'roi', 'roe',
        'roa', 'eps', 'pe', 'pb', 'dividend yield', 'market cap', 'volatility',
        'beta', 'alpha', 'correlation', 'diversification', 'allocation', 'strategy',
        'explain', 'concept', 'decision', 'making', 'financial decision', 'financial analysis',
        'financial planning', 'financial management', 'financial performance', 'financial metrics'
    ]
    
    text_lower = text.lower()
    
    # Check for exact matches first
    exact_matches = sum(1 for keyword in llama2_keywords if keyword in text_lower)
    
    # Check for partial matches (words that contain llama2-related terms)
    partial_matches = sum(1 for word in text_lower.split() if any(keyword in word for keyword in llama2_keywords))
    
    # Calculate confidence score
    # Give more weight to exact matches
    confidence = min((exact_matches * 0.7 + partial_matches * 0.3) / 5, 1.0)
    
    # Lower the threshold for llama2-related content
    return confidence > 0.1, confidence

def get_model_name(size: str) -> str:
    """Get the appropriate Llama 2 model name based on size."""
    return f"llama2:{size}"

@app.get("/")
async def root():
    return {"message": "llama2 Document Analysis API is running"}

@app.get("/models")
async def list_models():
    """List available models and their status."""
    try:
        models = ollama.list()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    try:
        print(f"Received request with text: {request.text[:100]}...")  # Log first 100 chars
        
        # Check if the text is llama2-related
        is_llama2, confidence = is_llama2_related(request.text)
        print(f"llama2 check - is_llama2: {is_llama2}, confidence: {confidence}")
        
        if not is_llama2:
            return AnalysisResponse(
                summary="This query appears to be outside the llama2 domain. This model is specialized in llama2-related content only.",
                roadmap="N/A - Content is not llama2-related",
                key_concepts=[],
                difficulty_level="N/A",
                is_llama2_domain=False,
                domain_confidence=confidence
            )

        model_name = get_model_name(request.model_size)
        print(f"Using model: {model_name}")
        
        try:
            # Generate summary with llama2-specific focus
            print("Generating summary...")
            summary_prompt = f"""<s>[INST] You are a llama2 domain expert using the latest Llama 2 model. Analyze the following financial text and provide a comprehensive analysis. The text is of type: {request.contentType}. Please follow these steps:

1. First, identify and list all different types of financial content in the text
2. Then, for each type of content:
   - Provide a detailed overview
   - Highlight the main financial points and key information
   - Note any important financial details or requirements
   - Identify underlying financial themes and patterns
3. Finally, provide an overall summary that ties everything together
4. If advanced analysis is requested, also include:
   - Key financial concepts and their relationships
   - Difficulty level assessment
   - Prerequisites for understanding the financial content

Text content:
{request.text}

Please structure your response as follows:
1. Financial Content Types Found:
   - [List all types of financial content found]

2. Detailed Financial Analysis:
   [For each content type, provide its summary]

3. Overall Financial Summary:
   [Provide a comprehensive summary that covers all financial content]

4. Advanced Financial Analysis (if requested):
   - Key Financial Concepts: [List main concepts]
   - Difficulty Level: [Assess complexity]
   - Prerequisites: [List required financial knowledge] [/INST]"""

            summary_response = ollama.generate(
                model=model_name,
                prompt=summary_prompt,
                options={
                    'num_predict': 2000,
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'top_k': 50,
                    'repeat_penalty': 1.2,
                    'presence_penalty': 0.1,
                    'frequency_penalty': 0.1
                }
            )
            print("Summary generated successfully")

            # Generate llama2-specific roadmap
            print("Generating roadmap...")
            roadmap_prompt = f"""<s>[INST] You are a llama2 education expert using the latest Llama 2 model. Based on the following financial text, create a detailed learning roadmap. The text is of type: {request.contentType}. Please:

1. First, perform a comprehensive financial content analysis:
   - Identify all financial topics and subtopics
   - Assess complexity levels
   - Determine financial prerequisites
   - Identify key financial learning objectives
2. Then, create an advanced learning path that:
   - Starts with foundational financial concepts
   - Progresses through different financial content types
   - Includes financial practice opportunities and assessments
   - Incorporates all types of financial content
   - Suggests additional financial resources
3. Finally, provide a detailed study schedule with:
   - Time estimates for each financial section
   - Recommended financial study methods
   - Financial milestone checkpoints
   - Progress tracking suggestions

Text content:
{request.text}

Please structure your response as follows:
1. Financial Content Analysis:
   [List and describe each type of financial content]

2. Financial Learning Roadmap:
   [Provide a detailed, step-by-step financial learning path]

3. Financial Study Schedule:
   [Suggest a timeline with financial milestones]

4. Additional Financial Resources:
   [List recommended financial supplementary materials] [/INST]"""

            roadmap_response = ollama.generate(
                model=model_name,
                prompt=roadmap_prompt,
                options={
                    'num_predict': 2000,
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'top_k': 50,
                    'repeat_penalty': 1.2,
                    'presence_penalty': 0.1,
                    'frequency_penalty': 0.1
                }
            )
            print("Roadmap generated successfully")

        except Exception as e:
            print(f"Error during model generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during model generation: {str(e)}")

        # Parse the response to extract additional information
        response_text = summary_response['response']
        key_concepts = []
        difficulty_level = "Medium"

        if request.advanced_analysis:
            if "Key Financial Concepts:" in response_text:
                concepts_section = response_text.split("Key Financial Concepts:")[1].split("\n")[0]
                key_concepts = [c.strip() for c in concepts_section.split(",")]
            
            if "Difficulty Level:" in response_text:
                difficulty_section = response_text.split("Difficulty Level:")[1].split("\n")[0]
                difficulty_level = difficulty_section.strip()

        return AnalysisResponse(
            summary=summary_response['response'],
            roadmap=roadmap_response['response'],
            key_concepts=key_concepts if request.advanced_analysis else None,
            difficulty_level=difficulty_level if request.advanced_analysis else None,
            is_llama2_domain=True,
            domain_confidence=confidence
        )

    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check if Llama 2 model is available
    try:
        models = ollama.list()
        llama_models = [m for m in models['models'] if m['name'].startswith('llama2')]
        if not llama_models:
            print("Warning: No Llama 2 models found. Please pull a model using: ollama pull llama2:7b")
    except Exception as e:
        print(f"Error checking models: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 
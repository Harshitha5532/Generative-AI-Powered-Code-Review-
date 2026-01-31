from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from groq import Groq
import re
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Code Review & Rewrite Agent")

# Configure Groq client
groq_client = Groq(api_key="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")

# Model configuration
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 4096
TOP_P = 0.9

# Pydantic models for request/response
class CodeReviewRequest(BaseModel):
    code: str
    language: str
    focus_areas: List[str] = []

class IssueDetail(BaseModel):
    line: Optional[int]
    severity: str
    category: str
    description: str
    suggestion: str

class CodeReviewResponse(BaseModel):
    status: str
    summary: str
    issues: List[IssueDetail]
    issue_counts: dict
    suggestions: str

class CodeRewriteRequest(BaseModel):
    code: str
    language: str
    focus_areas: List[str] = []

class CodeRewriteResponse(BaseModel):
    status: str
    rewritten_code: str
    improvements: str
    summary: str


def parse_review_response(review_text: str) -> dict:
    """
    Parse the LLM-generated review text into structured data
    """
    # Initialize counters
    issue_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    # Count issues by severity
    critical_matches = re.findall(r'\*\*Critical Issues?\*\*', review_text, re.IGNORECASE)
    high_matches = re.findall(r'\*\*High Priority\*\*', review_text, re.IGNORECASE)
    medium_matches = re.findall(r'\*\*Medium Priority\*\*', review_text, re.IGNORECASE)
    low_matches = re.findall(r'\*\*Low Priority\*\*', review_text, re.IGNORECASE)
    
    # Extract issue sections
    issues = []
    
    # Parse critical issues
    critical_section = re.search(r'\*\*Critical Issues?\*\*(.*?)(?=\*\*|$)', review_text, re.DOTALL | re.IGNORECASE)
    if critical_section:
        critical_items = re.findall(r'Line (\d+):(.*?)(?=Line \d+:|$)', critical_section.group(1), re.DOTALL)
        for line_num, desc in critical_items:
            issues.append({
                "line": int(line_num.strip()),
                "severity": "Critical",
                "category": "Bug/Security",
                "description": desc.strip()[:200],
                "suggestion": "Immediate fix required"
            })
            issue_counts["critical"] += 1
    
    # Parse high priority issues
    high_section = re.search(r'\*\*High Priority\*\*(.*?)(?=\*\*|$)', review_text, re.DOTALL | re.IGNORECASE)
    if high_section:
        high_items = re.findall(r'Line (\d+):(.*?)(?=Line \d+:|$)', high_section.group(1), re.DOTALL)
        for line_num, desc in high_items:
            issues.append({
                "line": int(line_num.strip()),
                "severity": "High",
                "category": "Performance/Logic",
                "description": desc.strip()[:200],
                "suggestion": "Should be addressed soon"
            })
            issue_counts["high"] += 1
    
    # Parse medium priority issues
    medium_section = re.search(r'\*\*Medium Priority\*\*(.*?)(?=\*\*|$)', review_text, re.DOTALL | re.IGNORECASE)
    if medium_section:
        medium_items = re.findall(r'Line (\d+):(.*?)(?=Line \d+:|$)', medium_section.group(1), re.DOTALL)
        for line_num, desc in medium_items:
            issues.append({
                "line": int(line_num.strip()),
                "severity": "Medium",
                "category": "Code Quality",
                "description": desc.strip()[:200],
                "suggestion": "Improvement recommended"
            })
            issue_counts["medium"] += 1
    
    # Parse low priority issues
    low_section = re.search(r'\*\*Low Priority\*\*(.*?)(?=\*\*|$)', review_text, re.DOTALL | re.IGNORECASE)
    if low_section:
        low_items = re.findall(r'Line (\d+):(.*?)(?=Line \d+:|$)', low_section.group(1), re.DOTALL)
        for line_num, desc in low_items:
            issues.append({
                "line": int(line_num.strip()),
                "severity": "Low",
                "category": "Style/Formatting",
                "description": desc.strip()[:200],
                "suggestion": "Optional enhancement"
            })
            issue_counts["low"] += 1
    
    return {
        "issues": issues,
        "issue_counts": issue_counts
    }


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    Serve the main HTML page
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index page not found")


@app.post("/api/review", response_model=CodeReviewResponse)
async def review_code(request: CodeReviewRequest):
    """
    Review code and provide detailed feedback
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Format focus areas
    focus_text = ", ".join(request.focus_areas) if request.focus_areas else "bugs, security, performance, and best practices"
    
    # Build the prompt for code review
    prompt = f"""You are an expert code reviewer. Analyze the following {request.language} code and provide a detailed review.

Focus areas: {focus_text}

Code to review:
```{request.language}
{request.code}
```

Provide your review in the following format:

**Summary**
A brief overview of the code quality and main findings.

**Critical Issues** (bugs, security vulnerabilities)
Line X: [Description of critical issue]
Line Y: [Description of another critical issue]

**High Priority** (performance issues, logic errors)
Line X: [Description of high priority issue]

**Medium Priority** (code quality, maintainability)
Line X: [Description of medium priority issue]

**Low Priority** (style, minor improvements)
Line X: [Description of low priority issue]

**Overall Suggestions**
General recommendations for improvement.

Be specific, reference line numbers where possible, and provide actionable feedback."""

    try:
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert code reviewer with deep knowledge of software engineering best practices, security, and performance optimization."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P
        )
        
        review_text = response.choices[0].message.content
        
        # Parse the review response
        parsed_data = parse_review_response(review_text)
        
        # Extract summary and suggestions
        summary_match = re.search(r'\*\*Summary\*\*(.*?)(?=\*\*|$)', review_text, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(1).strip() if summary_match else "Code review completed."
        
        suggestions_match = re.search(r'\*\*Overall Suggestions?\*\*(.*?)$', review_text, re.DOTALL | re.IGNORECASE)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else review_text
        
        return CodeReviewResponse(
            status="success",
            summary=summary,
            issues=parsed_data["issues"],
            issue_counts=parsed_data["issue_counts"],
            suggestions=suggestions if suggestions else review_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during code review: {str(e)}")


@app.post("/api/rewrite", response_model=CodeRewriteResponse)
async def rewrite_code(request: CodeRewriteRequest):
    """
    Rewrite and optimize code
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Format focus areas
    focus_text = ", ".join(request.focus_areas) if request.focus_areas else "optimization, best practices, and readability"
    
    # Build the prompt for code rewriting
    prompt = f"""You are an expert software developer. Rewrite and optimize the following {request.language} code.

Focus on: {focus_text}

Original code:
```{request.language}
{request.code}
```

Provide:
1. **Rewritten Code**: The complete, optimized code with improvements
2. **Key Improvements**: A bulleted list of the main changes made
3. **Summary**: A brief explanation of the optimization approach

Format your response as:

**Rewritten Code**
```{request.language}
[Your optimized code here]
```

**Key Improvements**
- [Improvement 1]
- [Improvement 2]
- [Improvement 3]

**Summary**
[Brief explanation of the changes and their benefits]

Ensure the rewritten code is clean, well-documented, production-ready, and follows best practices for {request.language}."""

    try:
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert software developer specializing in code optimization, refactoring, and best practices."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P
        )
        
        result_text = response.choices[0].message.content
        
        # Extract rewritten code
        code_match = re.search(r'```(?:' + re.escape(request.language) + r')?\s*(.*?)```', result_text, re.DOTALL)
        rewritten_code = code_match.group(1).strip() if code_match else request.code
        
        # Extract improvements
        improvements_match = re.search(r'\*\*Key Improvements?\*\*(.*?)(?=\*\*|$)', result_text, re.DOTALL | re.IGNORECASE)
        improvements = improvements_match.group(1).strip() if improvements_match else "Code has been optimized."
        
        # Extract summary
        summary_match = re.search(r'\*\*Summary\*\*(.*?)$', result_text, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(1).strip() if summary_match else "Code rewritten successfully."
        
        return CodeRewriteResponse(
            status="success",
            rewritten_code=rewritten_code,
            improvements=improvements,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during code rewrite: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model": MODEL_NAME}
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)




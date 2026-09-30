from risk_engine import calculate_priority
from gemini_service import generate_advisory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cyclone Impact Forecaster API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Cyclone Impact Forecaster API",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/api/advisory")
def advisory(request: dict):
    risk_data = request["risk_data"]

    infrastructure = risk_data.get("infrastructure", [])

    prioritized = calculate_priority(infrastructure)

    enriched_risk_data = {
        **risk_data,
        "infrastructure": prioritized
    }

    advisory_text = generate_advisory(enriched_risk_data)

    return {
        "risk_data": enriched_risk_data,
        "advisory": advisory_text,
        "priorities": prioritized
    }
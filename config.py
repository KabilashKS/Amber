import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Available Models Registry (High-performance multi-model support via NVIDIA NIM & Google Cloud)
AVAILABLE_MODELS = {
    # Flagship Meta & DeepSeek (NVIDIA NIM)
    "meta_muse": {
        "name": "Muse 30B",
        "provider": "nvidia",
        "model": "openai/meta/muse-glimmer-30b",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Fast",
        "context": "Fast · Quick creative & technical research"
    },
    "nvidia_llama": {
        "name": "Llama 3.2 11B Vision",
        "provider": "nvidia",
        "model": "openai/meta/llama-3.2-11b-vision-instruct",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Light",
        "context": "Light · Multimodal vision & efficient retrieval"
    },
    "nvidia_deepseek": {
        "name": "DeepSeek V4.1 Flash",
        "provider": "nvidia",
        "model": "openai/deepseek-ai/deepseek-v4.1-flash",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Medium",
        "context": "Medium · Complex logic & scientific deduction"
    },
    # NVIDIA Nemotron Family
    "nvidia_nemotron_lightning": {
        "name": "Nemotron 30B",
        "provider": "nvidia",
        "model": "openai/nvidia/nemotron-3.5-lightning-30b-a3b",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Fast",
        "context": "Fast · High-speed real-time search & synthesis"
    },
    "nvidia_nemotron_ultra": {
        "name": "Nemotron 550B Ultra",
        "provider": "nvidia",
        "model": "openai/nvidia/nemotron-3-ultra-550b-a55b",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Heavy",
        "context": "Heavy · Frontier enterprise research intelligence"
    },
    "nvidia_nemotron_super": {
        "name": "Nemotron 120B Super",
        "provider": "nvidia",
        "model": "openai/nvidia/nemotron-3-super-120b-a12b",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Medium",
        "context": "Medium · Balanced multi-domain analysis"
    },
    # Vision, Multimodal & Specialty Models (NVIDIA NIM)
    "meta_llama_vision": {
        "name": "Llama 3.2 11B Vision",
        "provider": "nvidia",
        "model": "openai/meta/llama-3.2-11b-vision-instruct",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Light",
        "context": "Light · Vision & multimodal low-latency reasoning"
    },
    "google_gemma4": {
        "name": "Gemma 4 31B IT",
        "provider": "nvidia",
        "model": "openai/google/gemma-4-31b-it",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Medium",
        "context": "Medium · High-accuracy reasoning & citations"
    },
    "moonshot_kimi": {
        "name": "Moonshot Kimi K3",
        "provider": "nvidia",
        "model": "openai/moonshotai/kimi-k3",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Heavy",
        "context": "Heavy · Massive context & deep literature memory"
    },
    "openai_gpt_oss": {
        "name": "GPT-OSS 20B",
        "provider": "nvidia",
        "model": "openai/openai/gpt-oss-20b",
        "base_url": NVIDIA_BASE_URL,
        "api_key_env": "NVIDIA_API_KEY",
        "tier": "Light",
        "context": "Light · Open architecture knowledge core"
    },
    # Google Cloud
    "gemini": {
        "name": "Gemini 1.5 Flash",
        "provider": "google",
        "model": "gemini/gemini-1.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "tier": "Fast",
        "context": "Fast · Cloud sub-second real-time streaming"
    }
}

def get_llm(model_id: str = "nvidia_nemotron_lightning") -> LLM:
    """Returns the configured CrewAI LLM for the requested model ID."""
    cfg = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS["nvidia_nemotron_lightning"])
    
    if cfg["provider"] == "nvidia":
        api_key = os.getenv(cfg["api_key_env"]) or os.getenv("NVIDIA_API_KEY", "")
        model_name = cfg["model"]
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        return LLM(
            model=model_name,
            base_url=cfg["base_url"],
            api_key=api_key,
            temperature=0.2
        )
    elif cfg["provider"] == "google":
        api_key = os.getenv(cfg["api_key_env"]) or os.getenv("GEMINI_API_KEY", "")
        return LLM(
            model=cfg["model"],
            api_key=api_key
        )
    
    # Fallback default
    return LLM(
        model=AVAILABLE_MODELS["nvidia_nemotron_lightning"]["model"],
        base_url=NVIDIA_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY", "")
    )

# Default LLM initialized on start for agent modules
master_llm = get_llm("nvidia_nemotron_lightning")

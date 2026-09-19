import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768 #all-mpnet-base-v2

_active_model = None
_model_type: str|None = None

def probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.GEMINI_API_KEY,
            model= "models/gemini-embedding-2-preview"
        )
        model.embed_query("test")
        logfire.info(f"Gemini embeding ready  gemini-embedding-2-preview")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. we will use transformer fallback")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading fallback embedding model: all-mpnet-base-v2")
    model = SentenceTransformer("all-mpnet-base-v2")
    return model

def _init():
    global _active_model, _model_type

    if _active_model is not None:
        return

    gemini_model = probe_gemini()

    if gemini_model is not None:
        _active_model = gemini_model
        _model_type = "gemini"
        
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"

def get_embedding_dim()-> int:
    """ Return the vector dimension for the active model. call afte _init()."""
    _init()
    

    if _model_type == "gemini":
        return _GEMINI_DIM
    else:
        return _FALLBACK_DIM


def _embed_batch(batch: list[str])->list[list[float]] :
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ["rate limit", "quota exceeded", "429"])
                if is_rate_limit and attempt <3:
                    wait =  2**attempt
                    logfire.warning(f"Gemini rate limit error, retrying in {wait} seconds: {e} attempt {attempt+1}/4")
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embed failed: {e}")
                    raise
        raise RuntimeError("Gemini embed failed after 4 attempts")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()
    

def embed_query(query:str)-> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    else:
        return _active_model.encode([query], show_progress_bar=False).tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embedings : list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]         
        with logfire.span(f"embed_batch ", model = _model_type, start= i , size= len(batch)):
             all_embedings.extend( _embed_batch(batch))
    return all_embedings
  
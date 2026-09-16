"""LLM Client abstraction for offline LLM integration (Ollama)."""

import re
import signal
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        pass


class OllamaClient(LLMClient):
    """Ollama client for offline LLM inference."""
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3", "llama3.2", "mistral", "codellama")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None
        self._available = False
        
        if OLLAMA_AVAILABLE:
            try:
                # Test connection
                self._client = ollama.Client(host=base_url, timeout=timeout)
                # Try to list models to verify connection
                try:
                    models_response = self._client.list()
                    model_names = []
                    
                    # Handle different response formats
                    # Ollama returns a Models object with .models attribute
                    if hasattr(models_response, 'models'):
                        model_list = models_response.models
                    elif isinstance(models_response, dict) and 'models' in models_response:
                        model_list = models_response['models']
                    elif isinstance(models_response, list):
                        model_list = models_response
                    else:
                        model_list = []
                    
                    # Extract model names
                    for m in model_list:
                        if hasattr(m, 'model'):
                            name = m.model
                        elif hasattr(m, 'name'):
                            name = m.name
                        elif isinstance(m, dict):
                            name = m.get('name') or m.get('model') or m.get('id', '')
                        else:
                            name = str(m)
                        if name:
                            model_names.append(name)
                    
                    self._available = True
                    if model_names:
                        print(f"[OK] Ollama connected. Available models: {[m.split(':')[0] for m in model_names[:3]]}")
                    else:
                        print(f"[OK] Ollama connected (no models found). Pull a model with: ollama pull {model}")
                except Exception as e:
                    # Try a simple test generation instead
                    try:
                        test_response = self._client.generate(model=self.model, prompt="test")
                        self._available = True
                        print(f"[OK] Ollama connected (model: {self.model})")
                    except Exception as e2:
                        print(f"[WARN] Ollama connection issue: {e2}")
                        print("  Make sure Ollama is running: ollama serve")
                        self._available = False
            except Exception as e:
                print(f"[WARN] Ollama initialization failed: {e}")
                self._available = False
        else:
            print("[WARN] Ollama package not installed. Install with: pip install ollama")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available and OLLAMA_AVAILABLE
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None
    ) -> str:
        """Generate text using Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama is not available. Make sure it's installed and running.")
        
        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Use SIGALRM to enforce a hard total-time timeout on the LLM call.
            # The httpx timeout in ollama.Client is per-chunk (read timeout),
            # not total — a slow-streaming response can hang indefinitely.
            def _alarm_handler(signum, frame):
                raise TimeoutError("Ollama LLM call exceeded total timeout")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.timeout)
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options=options
                )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # ollama >= 0.4 returns a ChatResponse object, not a dict
            msg = response.message if hasattr(response, "message") else response["message"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            return content.strip()

        except TimeoutError:
            raise RuntimeError(
                f"Ollama generation timed out after {self.timeout}s. "
                f"The model may be stuck on a complex prompt."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def generate_streaming(self, prompt: str, temperature: float = 0.7):
        """Generate text with streaming (for long outputs)."""
        if not self.is_available():
            raise RuntimeError("Ollama is not available.")
        
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
                stream=True
            )
            
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}")


class PlaceholderClient(LLMClient):
    """Placeholder LLM that returns dummy responses. Use for testing without Ollama."""

    def __init__(self):
        pass

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> str:
        # Return minimal valid responses so the pipeline can run
        if "RESEARCH_QUESTION:" in prompt or "general idea" in prompt.lower():
            return (
                "RESEARCH_QUESTION: How do trust and well-being relate across countries?\n"
                "CORE_CONSTRUCTS: trust, well-being, political attitudes\n"
                "HYPOTHESIZED_RELATIONS: Trust positively associated with well-being; moderation by country.\n"
                "CANDIDATE_CONFOUNDERS: age, education, country, gender\n"
                "DATA_REQUIREMENTS: ordinal scales, country fixed effects, weights"
            )
        if "hypothesis" in prompt.lower() and ("numbered" in prompt.lower() or "1." in prompt or "per line" in prompt.lower()):
            return (
                "1. Q58 is positively associated with Q238 controlling for B_COUNTRY.\n"
                "2. Q60 is positively correlated with Q249 by country.\n"
                "3. I_TRUSTCOURTS negatively correlates with Q238 when controlling for education."
            )
        # Minimal valid Python for CoderRunner so one round can complete
        if "def test_hypothesis" in prompt or "test_hypothesis(df)" in prompt or "return {" in prompt:
            return '''def test_hypothesis(df):
    import pandas as pd
    return {"effect_size": 0.1, "p_value": 0.05, "n_observations": len(df), "coverage": 1.0, "confidence_interval": (0.0, 0.2)}'''
        return "Placeholder response (set llm.provider to 'ollama' and start Ollama for real runs)."


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    Create an LLM client based on configuration.
    
    Args:
        config: LLM configuration dictionary
        
    Returns:
        LLMClient instance
        
    Raises:
        RuntimeError: If LLM provider is not available or configuration is invalid
        ValueError: If provider is unknown
    """
    provider = config.get("provider", "ollama").lower()
    
    if provider == "ollama":
        model = config.get("model", "llama3")
        base_url = config.get("base_url", "http://localhost:11434")
        timeout = config.get("timeout", 300)
        
        client = OllamaClient(model=model, base_url=base_url, timeout=timeout)
        
        if not client.is_available():
            raise RuntimeError(
                f"Ollama is not available. Please ensure:\n"
                f"  1. Ollama is installed: https://ollama.ai\n"
                f"  2. Ollama service is running: ollama serve\n"
                f"  3. Model '{model}' is available: ollama pull {model}\n"
                f"  4. Ollama is accessible at {base_url}"
            )
        
        return client

    if provider == "placeholder":
        print("[INFO] Using placeholder LLM (no Ollama). Set llm.provider to 'ollama' for real runs.")
        return PlaceholderClient()

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Supported providers: 'ollama', 'placeholder'. "
        f"Please check your config.yaml 'llm.provider' setting."
    )


def parse_list_response(text: str, num_items: Optional[int] = None) -> List[str]:
    """
    Parse a list response from LLM (e.g., numbered list, bullet points).
    
    Args:
        text: LLM response text
        num_items: Expected number of items (optional)
        
    Returns:
        List of extracted items
    """
    items = []
    
    # Try numbered list (1., 2., etc.)
    pattern = r'^\d+[\.\)]\s*(.+)$'
    matches = re.findall(pattern, text, re.MULTILINE)
    if matches:
        items = [m.strip() for m in matches]
    
    # Try bullet points (-, *, •)
    if not items:
        pattern = r'^[-*•]\s*(.+)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            items = [m.strip() for m in matches]
    
    # Try simple line breaks (one item per line)
    if not items:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Filter out lines that look like instructions or metadata
        items = [line for line in lines if not line.startswith(('Note:', 'Format:', 'Example:', 'Remember:'))]
    
    # Limit to requested number
    if num_items and len(items) > num_items:
        items = items[:num_items]
    
    return items


# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "upstage/Llama-2-70b-instruct-v2"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(TOKEN_CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
        )

    def predict(
        self,
        prompt: str = Input(description="Your prompt", default="Thomas is healthy, but he has to go to the hospital. What could be the reasons?"),
        new_tokens: int = Input(description="Generate at most this many new tokens in the response", ge=0, le=9216, default=1024),
    ) -> str:
        """Run a single prediction on the model"""
        full_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        del inputs["token_type_ids"]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        outputs = self.model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=new_tokens)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return output

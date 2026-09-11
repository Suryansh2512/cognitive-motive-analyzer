"""Llama 3.1 inference entrypoint for the motive analyzer."""

from src.model.llama_inference import analyze, _load_model

__all__ = ["analyze", "_load_model"]


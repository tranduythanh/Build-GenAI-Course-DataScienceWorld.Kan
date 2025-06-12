from __future__ import annotations

import base64
import io
import os
from typing import Any

import openai
from crewai.tools import tool
from PIL import Image

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception:  # pragma: no cover - azure may not be installed in tests
    speechsdk = None


@tool("Analyze TOEIC picture")
def analyze_picture(image_path: str) -> str:
    """Describe the picture using gpt-4o vision if credentials exist."""
    with Image.open(image_path) as img:
        width, height = img.size
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"Picture size: {width}x{height}px"

    img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    client = openai.Client(api_key=api_key)
    messages: Any = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the TOEIC-style picture for speaking practice.",  # noqa: E501
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_b64}",  # noqa: E501
                },
            ],
        }
    ]
    response = client.chat.completions.create(  # type: ignore[misc,call-arg]
        model="gpt-4o",
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


@tool("Generate sample description")
def generate_sample_description(info: str) -> str:
    """Create a simple description for a picture"""
    return f"This picture shows a scene {info.lower()} with basic elements."


@tool("Score student answer")
def assess_description(sample: str, user_answer: str) -> str:
    """Score user description based on overlap with sample text"""
    sample_words = set(sample.lower().split())
    user_words = set(user_answer.lower().split())
    if not sample_words:
        return "Score: 0"
    overlap = len(sample_words & user_words)
    score = int(min(100, (overlap / len(sample_words)) * 100))
    return f"Score: {score}"


@tool("Provide learning feedback")
def provide_feedback(score_text: str) -> str:
    """Return improvement advice based on score"""
    try:
        score = int(score_text.split(":")[1])
    except (IndexError, ValueError):
        score = 0
    if score >= 80:
        return "Great job! Proceed to the next picture."
    return "Keep practicing. Focus on describing key details."


@tool("Transcribe audio with pronunciation assessment")
def transcribe_audio(audio_path: str) -> str:
    """Return transcript and pronunciation score using Azure Speech."""
    if speechsdk is None:
        return "Azure SDK not installed"

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        return "Missing Azure credentials"

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

    pa_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,  # noqa: E501
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
    )
    pa_config.apply_to(recognizer)

    result = recognizer.recognize_once()
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        return "Recognition failed"
    pa_result = speechsdk.PronunciationAssessmentResult(result)
    return f"{result.text} | Pronunciation score: {pa_result.pronunciation_score}"  # noqa: E501

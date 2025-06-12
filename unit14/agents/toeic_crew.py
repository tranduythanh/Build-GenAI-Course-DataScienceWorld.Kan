from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..db import init_db, log_session
from .tools import (
    analyze_picture,
    assess_description,
    generate_sample_description,
    provide_feedback,
    transcribe_audio,
)


def build_crew() -> Crew:
    """Create crew with four specialized agents."""
    analyzer = Agent(
        role="Picture Analysis Agent - Image Analysis Specialist",
        goal=(
            "Provide deep structural breakdowns of TOEIC images, listing "
            "objects, layout and key vocabulary to support Band 7 mastery"
        ),
        backstory=(
            "A computer vision expert trained on a curated TOEIC picture "
            "database. Delivers multi-angle analyses and detailed coaching "
            "points for intensive practice"
        ),
        tools=[analyze_picture],
        llm="gpt-4o",
        verbose=False,
    )

    writer = Agent(
        role="Sample Writing Agent - TOEIC Writing Specialist",
        goal=(
            "Create multiple TOEIC-style sample descriptions using the "
            "analysis results with varying difficulty levels"
        ),
        backstory=(
            "An academic writing coach well versed in TOEIC Speaking "
            "structures and vocabulary at the 150-180 range, supplying "
            "reference answers"
        ),
        tools=[generate_sample_description],
        llm="gpt-4o-mini",
        verbose=False,
    )

    assessor = Agent(
        role="Assessment Agent - Audio Analysis & Scoring Specialist",
        goal=(
            "Generate detailed TOEIC Speaking scores with JSON output "
            "covering pronunciation, grammar, vocabulary and fluency"
        ),
        backstory=(
            "Integrates ASR engines and scoring algorithms to mimic "
            "official TOEIC standards. Uses simplified heuristics in this PoC"
        ),
        tools=[assess_description, transcribe_audio],
        llm="gpt-4o-mini",
        verbose=False,
    )

    mentor = Agent(
        role="Learning Mentor Agent - Personalized Coaching Specialist",
        goal="Deliver targeted practice plans and progress tracking to reach Band 7",  # noqa: E501
        backstory=(
            "Processes assessment data and historical records to group "
            "errors, prioritize frequent issues and design adaptive "
            "improvement strategies"
        ),
        tools=[provide_feedback],
        llm="gpt-4o-mini",
        verbose=False,
    )

    task1 = Task(
        description="Analyze the uploaded picture",
        expected_output="Picture info",
        agent=analyzer,
    )
    task2 = Task(
        description="Create a sample description",
        expected_output="Sample text",
        agent=writer,
    )
    task3 = Task(
        description="Assess the student's answer",
        expected_output="Score text",
        agent=assessor,
    )
    task4 = Task(
        description="Give learning feedback",
        expected_output="Advice",
        agent=mentor,
    )

    crew = Crew(
        agents=[analyzer, writer, assessor, mentor],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
    )
    return crew


def run_pipeline(image_path: str, user_answer: str) -> tuple[str, str, str]:
    """Run simple pipeline using agent tools directly and store results."""
    init_db()
    info = analyze_picture.run(image_path)
    sample = generate_sample_description.run(info)
    score = assess_description.run(sample, user_answer)
    feedback = provide_feedback.run(score)
    log_session(image_path, user_answer, None, sample, score, feedback)
    return sample, score, feedback


def run_audio_pipeline(
    image_path: str,
    audio_path: str,
) -> tuple[str, str, str, str]:
    """Return transcript, sample, score and feedback for a spoken answer."""
    init_db()
    info = analyze_picture.run(image_path)
    transcript = transcribe_audio.run(audio_path)
    sample = generate_sample_description.run(info)
    score = assess_description.run(sample, transcript)
    feedback = provide_feedback.run(score)
    log_session(image_path, audio_path, transcript, sample, score, feedback)
    return transcript, sample, score, feedback

import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from typing import List
from pydantic import BaseModel, Field, EmailStr

class ExoType(BaseModel):
    type: List[str] = Field(
        ...,
        description="Type d'exercice à générer, par exemple: ['QCM', 'Vrai/Faux', 'Texte à trous']",
    )

class SyntheseExo(BaseModel):
    n: int = Field(..., description="Nombre d'exercices à générer")
    difficulte: int = Field(..., description="Niveau de difficulté de l'exercice (1-5)")
    description: str = Field(..., description="Description du sujet de l'exercice")
    type: ExoType


def generateur_exo(SyntheseExo: SyntheseExo) -> dict:
    """Génère un exercice basé sur les paramètres fournis.

    Args:
        SyntheseExo (SyntheseExo): Les paramètres de l'exercice à générer.

    Returns:
        dict: status and result or error msg.
    """


    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


refinement_loop = LoopAgent(
    name="RefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        critic_agent_in_loop,
        refiner_agent_in_loop,
    ],
    max_iterations=5 # Limit loops
)

generateur_exo_agent = LlmAgent(
    name="generation_exo_agent",
    model="gemini-2.0-flash",
    description=(
        "generateur d'exo"
    ),
    instruction=(
        "Appelle l'outil generateur_exo"
    ),
    tools=[generateur_exo],
)


root_agent = SequentialAgent(
    name="IterativeWritingPipeline",
    sub_agents=[
        initial_writer_agent, 
        refinement_loop       
    ],
    description="Writes an initial document and then iteratively refines it with critique using an exit tool."
)
from pydantic import BaseModel
from typing import List

# --- MODÈLES D'ENTRÉE ---

class TimeSeriesPoint(BaseModel):
    annee: int
    capture: float
    effort: float

class EconomicVariables(BaseModel):
    prix_kg: float
    cout_effort: float

class SimulationRequest(BaseModel):
    historique: List[TimeSeriesPoint]
    economie: EconomicVariables
    modele_choisi: str = "schaefer" # "schaefer" ou "fox"

# --- MODÈLES DE SORTIE ---

class ReferencePoints(BaseModel):
    f_msy: float
    y_msy: float
    f_mey: float
    y_mey: float
    libre_acces_effort: float

class SimulationPoint(BaseModel):
    effort: float
    capture_estimee: float
    profit_estime: float

class SimulationResponse(BaseModel):
    interpretation_ia: str          # <-- AJOUT : pour l'interprétation du modèle IA
    selected_model: str            # <-- AJOUT : pour confirmer quel modèle est affiché
    r_squared: float              # <-- AJOUT : pour la validation du modèle (ex: 0.85)
    diagnostic: str               # ex: "Surexploitation", "Sous-exploité"
    points_reference: ReferencePoints
    points_simulation: List[SimulationPoint]

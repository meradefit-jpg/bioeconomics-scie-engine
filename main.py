from fastapi import FastAPI, HTTPException
from models.schemas import SimulationRequest, SimulationResponse
from services.calculator import run_bioeconomic_model

app = FastAPI(
    title="Plateforme Bioéconomique - Moteur Scientifique",
    description="Microservice Python dédié aux calculs d'évaluation halieutique",
    version="1.0.0"
)

@app.post("/api/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    # Sécurité basique : on vérifie qu'on a assez de données pour faire une régression
    if len(request.historique) < 3:
        raise HTTPException(status_code=400, detail="Au moins 3 points historiques sont requis pour l'analyse.")

    try:
        # Appel du service mathématique (qui renvoie désormais un dictionnaire complet)
        resultat = run_bioeconomic_model(
            historique=request.historique,
            eco=request.economie,
            modele=request.modele_choisi
        )
        
        # L'opérateur ** déballe le dictionnaire directement dans le modèle Pydantic
        # Cela associe automatiquement r_squared, diagnostic, etc., aux bons champs.
        return SimulationResponse(**resultat)
        
    except Exception as e:
        # Capture les erreurs mathématiques (ex: données incompatibles avec le log)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul : {str(e)}")
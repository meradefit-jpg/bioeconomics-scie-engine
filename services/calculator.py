import numpy as np
from scipy.optimize import fsolve
from models.schemas import TimeSeriesPoint, EconomicVariables, ReferencePoints, SimulationPoint
from services.ai_model import predire_interpretation_ia

def run_bioeconomic_model(historique: list[TimeSeriesPoint], eco: EconomicVariables, modele: str):
    # 1. Extraction et préparation des données
    efforts = np.array([p.effort for p in historique])
    captures = np.array([p.capture for p in historique])
    
    # Protection contre la division par zéro
    efforts_safe = np.where(efforts == 0, 1e-9, efforts)
    cpue = captures / efforts_safe
    
    p = eco.prix_kg * 1000
    c = eco.cout_effort

    # Variables pour stocker les prédictions (pour le calcul du R² linéaire)
    valeurs_reelles_regression = []
    valeurs_predites_regression = []

    # 2. Choix du modèle et Régressions
    if modele.lower() == "fox":
        # Modèle de Fox : ln(CPUE) = c_fox - d_fox * E
        ln_cpue = np.log(cpue)
        slope, intercept = np.polyfit(efforts, ln_cpue, 1)
        c_fox = intercept
        d_fox = -slope
        
        # Points de référence Biologiques (Fox)
        f_msy = 1 / d_fox
        y_msy = (1 / d_fox) * np.exp(c_fox - 1)
        
        def calc_capture_estimee(e):
            return e * np.exp(c_fox - (d_fox * e))

        # Points de référence Économiques (MEY) - Résolution numérique
        if p > 0:
            # Dérivée du profit pour Fox : dπ/dE = P * e^(c-dE) * (1-dE) - C = 0
            def deriv_profit(e):
                return p * np.exp(c_fox - d_fox * e) * (1 - d_fox * e) - c
            
            # fsolve trouve la racine (le point où la dérivée s'annule). Guess = f_msy * 0.5
            f_mey_opt = fsolve(deriv_profit, f_msy * 0.5)[0]
            f_mey = max(0, f_mey_opt)
        else:
            f_mey = 0
            
        y_mey = calc_capture_estimee(f_mey)
        
        # Libre accès (Profit = 0) -> Résolution algébrique exacte : E = (c_fox - ln(C/P)) / d_fox
        if p > 0 and c > 0:
            f_oa = (c_fox - np.log(c / p)) / d_fox
            f_oa = max(0, f_oa)
        else:
            f_oa = 0

        # Données pour R² (Modèle Linéaire Fox)
        valeurs_reelles_regression = ln_cpue
        valeurs_predites_regression = c_fox - (d_fox * efforts)

    else:
        # Modèle de Schaefer (Par défaut) : CPUE = a - b * E
        slope, intercept = np.polyfit(efforts, cpue, 1)
        a = intercept
        b = -slope # La pente est naturellement négative
        
        # Points de référence Biologiques (Schaefer)
        f_msy = a / (2 * b)
        y_msy = (a**2) / (4 * b)
        
        # Points de référence Économiques (Gordon-Schaefer)
        f_mey = (a - (c / p)) / (2 * b) if p > 0 else 0
        f_mey = max(0, f_mey) 
        y_mey = a * f_mey - b * (f_mey**2)
        
        # Libre accès (Profit = 0)
        f_oa = (a - (c / p)) / b if p > 0 else 0
        f_oa = max(0, f_oa)

        def calc_capture_estimee(e):
            return (a * e) - (b * (e**2))

        # Données pour R² (Modèle Linéaire Schaefer)
        valeurs_reelles_regression = cpue
        valeurs_predites_regression = a - (b * efforts)

    # 3. Validation Scientifique : Calcul du R² (Coefficient de détermination)
    ss_res = np.sum((valeurs_reelles_regression - valeurs_predites_regression) ** 2)
    ss_tot = np.sum((valeurs_reelles_regression - np.mean(valeurs_reelles_regression)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    r_squared = round(float(r_squared), 4)

    # 4. Assemblage des points de référence
    refs = ReferencePoints(
        f_msy=round(float(f_msy), 2),
        y_msy=round(float(y_msy), 2),
        f_mey=round(float(f_mey), 2),
        y_mey=round(float(y_mey), 2),
        libre_acces_effort=round(float(f_oa), 2)
    )

    # 5. Génération des points de simulation (pour le graphique Angular)
    max_effort = max(f_msy * 2.5, f_oa * 1.2)
    efforts_sim = np.linspace(0, max_effort, 50)
    
    points_sim = []
    for e in efforts_sim:
        y_est = calc_capture_estimee(e)
        if y_est < 0: y_est = 0
        
        profit = (p * y_est) - (c * e)
        points_sim.append(
            SimulationPoint(
                effort=round(float(e), 2),
                capture_estimee=round(float(y_est), 2),
                profit_estime=round(float(profit), 2)
            )
        )

    # 6. Phase de Diagnostic (Comparaison Effort actuel vs F_MSY)
    dernier_effort = efforts[-1]
    
    if dernier_effort < 0.8 * f_msy:
        diagnostic = "Sous-exploitation (Effort actuel bien inférieur au MSY)"
    elif 0.8 * f_msy <= dernier_effort <= 1.2 * f_msy:
        diagnostic = "Exploitation optimale (Effort actuel proche de l'équilibre MSY)"
    else:
        diagnostic = "Surexploitation (Effort actuel critique, supérieur au MSY)"

    # Calcul des variables requises pour le modèle de classification (sur l'effort ACTUEL)
    dernier_effort = float(efforts[-1])
    y_dernier = calc_capture_estimee(dernier_effort)
    if y_dernier < 0:
        y_dernier = 0.0
        
    dernier_profit = float((p * y_dernier) - (c * dernier_effort))
    cout_total_flotte = float(c * dernier_effort)

    output = {
        "selected_model": modele.lower(),
        "r_squared": r_squared,
        "diagnostic": diagnostic,
        "points_reference": refs,
        "points_simulation": points_sim
    }
    
    # Appel de la prédiction du modèle de Machine Learnig local
    output["interpretation_ia"] = predire_interpretation_ia(
        r_squared=r_squared,
        effort_actuel=dernier_effort,
        f_msy=f_msy,
        profit=dernier_profit,
        cout_total=cout_total_flotte
    )

    return output

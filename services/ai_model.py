import numpy as np
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# MODÈLE 1 : FIABILITÉ SCIENTIFIQUE (Modèle Statistique)
# Inputs : [R²]
# Outputs : 3 Niveaux de confiance statistique
# =====================================================================
X_train_r2 = np.array([
    [0.95], [0.88], [0.91], [0.84],  # Confiance Forte
    [0.71], [0.65], [0.55], [0.42],  # Confiance Modérée
    [0.30], [0.22], [0.11], [0.04]   # Alerte / Invalide
])
y_train_r2 = np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0])

DICTIONNAIRE_R2 = {
    2: "Validation Scientifique (IA) : L'ajustement mathématique du modèle est excellent. La corrélation entre l'effort de pêche et les captures historiques est hautement significative, ce qui confère une excellente fiabilité aux points de référence (MSY/MEY) pour guider les décisions.",
    1: "Validation Scientifique (IA) : L'ajustement mathématique est modéré. Le modèle explique de manière acceptable la dynamique du stock, mais l'analyse doit être complétée par une vigilance sur le terrain car des facteurs externes mineurs peuvent influencer les captures.",
    0: "Alerte Scientifique (IA) : La qualité de l'ajustement (R²) est statistiquement insuffisante. Les variations de captures ne dépendent pas uniquement de l'effort des navires. Les points de référence calculés doivent être manipulés avec une extrême réserve."
}

clf_r2 = DecisionTreeClassifier(random_state=42, max_depth=3)
clf_r2.fit(X_train_r2, y_train_r2)


# =====================================================================
# MODÈLE 2 : ÉTAT BIOLOGIQUE DE LA RESSOURCE (Modèle Halieutique)
# Inputs spécifiques : [Ratio_Effort (E/F_msy), Qualité_Modèle (R²)]
# -> L'IA évalue la pression sur la biomasse pondérée par la fiabilité du R².
# Outputs : 6 Niveaux de dégradation du stock (0 à 5)
# =====================================================================
X_train_msy = np.array([
    [0.25, 0.90], [0.35, 0.75], [0.15, 0.50], [0.39, 0.85],  # 0: Stock intact / Vierge
    [0.55, 0.88], [0.65, 0.70], [0.72, 0.95], [0.60, 0.60],  # 1: Exploitation modérée
    [0.90, 0.92], [1.00, 0.85], [1.05, 0.90], [0.95, 0.55],  # 2: Pleine exploitation (Seuil MSY)
    [1.15, 0.80], [1.25, 0.75], [1.30, 0.90], [1.22, 0.65],  # 3: Surexploitation débutante
    [1.45, 0.85], [1.60, 0.70], [1.70, 0.95], [1.50, 0.50],  # 4: Surexploitation sévère
    [1.95, 0.90], [2.20, 0.75], [2.50, 0.85], [2.05, 0.40]   # 5: Effondrement du stock
])
y_train_msy = np.array([
    0, 0, 0, 0,  
    1, 1, 1, 1,  
    2, 2, 2, 2,  
    3, 3, 3, 3,  
    4, 4, 4, 4,  
    5, 5, 5, 5   
])

DICTIONNAIRE_MSY = {
    0: "État de la Ressource (MSY) : Le stock de poissons est sous-exploité. La biomasse en mer est intacte et très proche de son état naturel d'origine.",
    1: "État de la Ressource (MSY) : Le stock subit une exploitation sécurisée. La biomasse se maintient confortablement au-dessus du seuil de renouvellement.",
    2: "État de la Ressource (MSY) : La pêcherie extrait la capture maximale soutenable. Le stock est exploité à sa limite biologique durable (Seuil MSY).",
    3: "État de la Ressource (MSY) : Signaux de surpêche biologique. La biomasse décline car le rythme des captures surpasse le taux de reproduction naturelle.",
    4: "État de la Ressource (MSY) : Le stock est sévèrement dégradé. On observe une diminution inquiétante de la taille moyenne des poissons capturés en mer.",
    5: "État de la Ressource (MSY) : Ruine biologique complète du capital naturel. Le stock est épuisé et la ressource est en situation d'effondrement critique."
}

clf_msy = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_msy.fit(X_train_msy, y_train_msy)


# =====================================================================
# MODÈLE 3 : PERFORMANCE FINANCIÈRE DE LA FLOTTE (Modèle Économique)
# Inputs spécifiques : [Ratio_Profit (Profit/Coût_Total), Marge_Nette (Profit/Recettes)]
# -> L'IA évalue la santé financière de la flotte sans regarder la biologie.
# Outputs : 6 États de santé financière (0 à 5)
# =====================================================================
X_train_mse = np.array([
    [0.60, 0.37], [0.50, 0.33], [0.70, 0.41], [0.55, 0.35],  # 0: Profits exceptionnels
    [0.22, 0.18], [0.20, 0.16], [0.25, 0.20], [0.18, 0.15],  # 1: Rentabilité maximale (MEY)
    [0.08, 0.07], [0.05, 0.04], [0.12, 0.10], [0.03, 0.02],  # 2: Profits positifs en baisse
    [0.00, 0.00], [0.01, 0.01], [-0.01, -0.01], [0.00, 0.00], # 3: Seuil de rentabilité nul (Libre Accès)
    [-0.15, -0.17], [-0.25, -0.33], [-0.12, -0.13], [-0.20, -0.25], # 4: Déficit d'exploitation modéré
    [-0.60, -1.50], [-0.80, -4.00], [-0.55, -1.22], [-0.75, -3.00]  # 5: Faillite sectorielle structurelle
])
y_train_mse = np.array([
    0, 0, 0, 0,  
    1, 1, 1, 1,  
    2, 2, 2, 2,  
    3, 3, 3, 3,  
    4, 4, 4, 4,  
    5, 5, 5, 5   
])

DICTIONNAIRE_MSE = {
    0: "Performance Économique (MSE) : L'activité génère un profit unitaire maximal par navire. La rareté de l'effort garantit des captures faciles à faible coût.",
    1: "Performance Économique (MSE) : La pêcherie atteint son point d'efficacité maximale. La rente financière globale de la flottille est optimisée au plus haut niveau (MEY).",
    2: "Performance Économique (MSE) : La rentabilité globale est en baisse. Une surcapacité de navires commence à diluer et diviser les bénéfices individuels.",
    3: "Performance Économique (MSE) : Dissipation totale de la rente économique (Libre Accès). Les revenus couvrent tout juste les charges d'exploitation.",
    4: "Performance Économique (MSE) : La flottille fonctionne à perte nette chronique. La raréfaction du poisson fait grimper en flèche les dépenses de carburant.",
    5: "Performance Économique (MSE) : Faillite économique structurelle. Les pertes financières sont colossales, ruinant la viabilité des entreprises de pêche."
}

clf_mse = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_mse.fit(X_train_mse, y_train_mse)


# =====================================================================
# MODÈLE 4 : ACTIONS POLITIQUES DE GESTION (Modèle Décisionnel)
# Inputs spécifiques : [Ratio_Effort (E/F_msy), Ratio_Profit (Profit/Coût_Total)]
# -> L'IA réalise un arbitrage multi-critères (Bio + Éco) pour choisir l'action.
# Outputs : 6 Directives politiques de régulation (0 à 5)
# =====================================================================
X_train_action = np.array([
    [0.35, 0.50], [0.45, 0.30], [0.20, 0.60], [0.50, 0.25],  # 0: Expansion autorisée
    [0.72, 0.22], [0.75, 0.20], [0.80, 0.15], [0.68, 0.24],  # 1: Fixation du MEY (Gel préventif)
    [0.95, 0.08], [1.00, 0.05], [1.05, 0.02], [0.90, 0.10],  # 2: Régulation par quotas (Seuil MSY)
    [1.25, 0.00], [1.35, -0.05], [1.20, 0.01], [1.15, 0.02], # 3: Réduction requise de l'effort
    [1.55, -0.15], [1.65, -0.25], [1.75, -0.35], [1.45, -0.10], # 4: Plan d'ajustement et restrictions
    [2.10, -0.60], [2.30, -0.80], [2.50, -1.10], [1.95, -0.50]  # 5: Fermeture d'urgence et moratoire
])
y_train_action = np.array([
    0, 0, 0, 0,  
    1, 1, 1, 1,  
    2, 2, 2, 2,  
    3, 3, 3, 3,  
    4, 4, 4, 4,  
    5, 5, 5, 5   
])

DICTIONNAIRE_ACTION = {
    0: "Action requise : Une expansion progressive et encadrée de l'effort de pêche (octroi prudent de nouvelles licences) est envisageable pour valoriser la ressource.",
    1: "Action requise : Stabilisation totale. Il est fortement recommandé de geler l'effort à son niveau actuel (numérus clausus strict) pour sanctuariser la rente économique.",
    2: "Action requise : Mesures de prévention. Il est interdit d'augmenter la taille de la flotte ou le nombre de sorties pour éviter de basculer vers la surpêche.",
    3: "Action requise : Réduction immédiate de l'effort de pêche. Il faut restreindre le temps de présence en mer pour reconstituer une marge économique viable.",
    4: "Action requise : Plan d'ajustement structurel obligatoire. Réduction drastique d'au moins 20% de la capacité de capture via des restrictions techniques strictes.",
    5: "Action requise : Mesures de sauvegarde radicales. Arrêt immédiat de toute activité commerciale (repos biologique total) pour éviter un désastre irréversible."
}

clf_action = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_action.fit(X_train_action, y_train_action)


# =====================================================================
# FONCTION UNIQUE D'INFÉRENCE MULTI-MODÈLES INDÉPENDANTS
# =====================================================================
def predire_interpretation_ia(r_squared: float, effort_actuel: float, f_msy: float, profit: float, cout_total: float) -> str:
    """
    Exécute 4 processus d'inférence parallèles. Chaque modèle extrait les variables 
    qui le concernent pour tracer ses propres frontières géométriques.
    """
    # 1. Modèle Statistique (Input: R² brut)
    X_test_r2 = np.array([[r_squared]])
    classe_r2 = int(clf_r2.predict(X_test_r2)[0])
    commentaire_r2 = DICTIONNAIRE_R2[classe_r2]
    
    # Préparation des indicateurs intermédiaires
    ratio_effort = float(effort_actuel / f_msy) if f_msy > 0 else 1.0
    ratio_profit = float(profit / cout_total) if cout_total > 0 else 0.0
    recettes = profit + cout_total
    marge_nette = float(profit / recettes) if recettes > 0 else -1.0
    
    # 2. Modèle Halieutique (Inputs: Ratio d'effort, R²)
    X_test_msy = np.array([[ratio_effort, r_squared]])
    classe_msy = int(clf_msy.predict(X_test_msy)[0])
    analyse_msy = DICTIONNAIRE_MSY[classe_msy]
    
    # 3. Modèle Économique (Inputs: Ratio de profit, Marge nette)
    X_test_mse = np.array([[ratio_profit, marge_nette]])
    classe_mse = int(clf_mse.predict(X_test_mse)[0])
    analyse_mse = DICTIONNAIRE_MSE[classe_mse]
    
    # 4. Modèle Décisionnel (Inputs: Ratio d'effort, Ratio de profit)
    X_test_action = np.array([[ratio_effort, ratio_profit]])
    classe_action = int(clf_action.predict(X_test_action)[0])
    analyse_action = DICTIONNAIRE_ACTION[classe_action]
    
    # Assemblage final du diagnostic multi-agents
    rapport_final = (
        f"{commentaire_r2}\n\n"
        f"Analyse Générée par l'IA :\n"
        f"• {analyse_msy}\n"
        f"• {analyse_mse}\n"
        f"• {analyse_action}"
    )
    
    return rapport_final
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# MODÈLE 1 : FIABILITÉ SCIENTIFIQUE
# =====================================================================
# (Les X_train et y_train restent strictement identiques)
X_train_r2 = np.array([[0.95], [0.88], [0.91], [0.84], [0.71], [0.65], [0.55], [0.42], [0.30], [0.22], [0.11], [0.04]])
y_train_r2 = np.array([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0])

# Nouvelles définitions avec balises dynamiques
DICTIONNAIRE_R2 = {
    2: "Validation Scientifique : Ajustement mathématique excellent ({r2}%). La corrélation historique garantit une haute fiabilité aux points de référence pour guider les décisions.",
    1: "Validation Scientifique : Ajustement mathématique modéré ({r2}%). Le modèle explique globalement la dynamique, mais l'analyse appelle à une vigilance sur le terrain.",
    0: "Alerte Scientifique : Qualité d'ajustement insuffisante ({r2}%). Les captures ne dépendent pas que de l'effort; les résultats doivent être manipulés avec une extrême réserve."
}
clf_r2 = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X_train_r2, y_train_r2)


# =====================================================================
# MODÈLE 2 : ÉTAT BIOLOGIQUE DE LA RESSOURCE
# =====================================================================
X_train_msy = np.array([[0.25, 0.90], [0.35, 0.75], [0.15, 0.50], [0.39, 0.85], [0.55, 0.88], [0.65, 0.70], [0.72, 0.95], [0.60, 0.60], [0.90, 0.92], [1.00, 0.85], [1.05, 0.90], [0.95, 0.55], [1.15, 0.80], [1.25, 0.75], [1.30, 0.90], [1.22, 0.65], [1.45, 0.85], [1.60, 0.70], [1.70, 0.95], [1.50, 0.50], [1.95, 0.90], [2.20, 0.75], [2.50, 0.85], [2.05, 0.40]])
y_train_msy = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

# Nouvelles définitions avec balises dynamiques
DICTIONNAIRE_MSY = {
    0: "État de la Ressource : Stock sous-exploité. L'effort actuel ({effort}) n'atteint que {ratio}% de la limite biologique ({msy}).",
    1: "État de la Ressource : Exploitation sécurisée. L'effort ({effort}) se maintient confortablement sous le seuil de renouvellement MSY ({msy}).",
    2: "État de la Ressource : Pleine exploitation. La flottille opère exactement à la limite biologique durable ({msy} navires).",
    3: "État de la Ressource : Surexploitation débutante. Avec {effort} unités, le rythme d'extraction surpasse le seuil de reproduction naturelle ({msy}).",
    4: "État de la Ressource : Surexploitation sévère. La flotte compte {diff} navires de trop par rapport au MSY, dégradant fortement la biomasse.",
    5: "État de la Ressource : Effondrement critique. L'effort massif ({effort} unités, soit {ratio}% du seuil toléré) ruine le capital naturel du stock."
}
clf_msy = DecisionTreeClassifier(random_state=42, max_depth=5).fit(X_train_msy, y_train_msy)


# =====================================================================
# MODÈLE 3 : PERFORMANCE FINANCIÈRE DE LA FLOTTE
# =====================================================================
X_train_mse = np.array([[0.60, 0.37], [0.50, 0.33], [0.70, 0.41], [0.55, 0.35], [0.22, 0.18], [0.20, 0.16], [0.25, 0.20], [0.18, 0.15], [0.08, 0.07], [0.05, 0.04], [0.12, 0.10], [0.03, 0.02], [0.00, 0.00], [0.01, 0.01], [-0.01, -0.01], [0.00, 0.00], [-0.15, -0.17], [-0.25, -0.33], [-0.12, -0.13], [-0.20, -0.25], [-0.60, -1.50], [-0.80, -4.00], [-0.55, -1.22], [-0.75, -3.00]])
y_train_mse = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

# Nouvelles définitions avec balises dynamiques
DICTIONNAIRE_MSE = {
    0: "Performance Économique : Profits exceptionnels. L'activité génère {profit} DZD de bénéfice net, soit une marge de {marge}%.",
    1: "Performance Économique : Rentabilité optimisée. La rente financière globale est maximisée au plus haut niveau (MEY).",
    2: "Performance Économique : Baisse de rentabilité. La surcapacité dilue les revenus individuels (Marge nette réduite à {marge}%).",
    3: "Performance Économique : Seuil de Libre Accès. Dissipation quasi-totale de la rente, les revenus couvrent tout juste les charges.",
    4: "Performance Économique : Déficit chronique. La flottille fonctionne à perte, creusant un déficit estimé à {abs_profit} DZD.",
    5: "Performance Économique : Faillite structurelle. Les pertes financières sont colossales ({abs_profit} DZD), ruinant la viabilité sectorielle."
}
clf_mse = DecisionTreeClassifier(random_state=42, max_depth=5).fit(X_train_mse, y_train_mse)


# =====================================================================
# MODÈLE 4 : ACTIONS POLITIQUES DE GESTION
# =====================================================================
X_train_action = np.array([[0.35, 0.50], [0.45, 0.30], [0.20, 0.60], [0.50, 0.25], [0.72, 0.22], [0.75, 0.20], [0.80, 0.15], [0.68, 0.24], [0.95, 0.08], [1.00, 0.05], [1.05, 0.02], [0.90, 0.10], [1.25, 0.00], [1.35, -0.05], [1.20, 0.01], [1.15, 0.02], [1.55, -0.15], [1.65, -0.25], [1.75, -0.35], [1.45, -0.10], [2.10, -0.60], [2.30, -0.80], [2.50, -1.10], [1.95, -0.50]])
y_train_action = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])

# Les actions restent prescriptives, donc pas de balises nécessaires
DICTIONNAIRE_ACTION = {
    0: "Action requise : Expansion progressive et encadrée (octroi prudent de nouvelles licences) envisageable.",
    1: "Action requise : Stabilisation totale (Gel préventif). Geler l'effort au niveau actuel pour sanctuariser la rente.",
    2: "Action requise : Régulation stricte (Seuil MSY). Interdiction absolue d'augmenter la taille de la flotte.",
    3: "Action requise : Réduction immédiate de l'effort. Restreindre le temps en mer pour reconstituer une marge viable.",
    4: "Action requise : Plan d'ajustement structurel. Réduction drastique requise (via des restrictions techniques ou retraits).",
    5: "Action requise : Mesures de sauvegarde radicales. Arrêt immédiat de l'activité (moratoire) pour éviter un désastre irréversible."
}
clf_action = DecisionTreeClassifier(random_state=42, max_depth=5).fit(X_train_action, y_train_action)


# =====================================================================
# FONCTION UNIQUE D'INFÉRENCE MULTI-MODÈLES
# =====================================================================
def predire_interpretation_ia(r_squared: float, effort_actuel: float, f_msy: float, profit: float, cout_total: float) -> str:
    # 1. Agent Statistique
    X_test_r2 = np.array([[r_squared]])
    classe_r2 = int(clf_r2.predict(X_test_r2)[0])
    
    # 2. Préparation des variables intermédiaires (avec gestion des coûts nuls)
    ratio_effort = float(effort_actuel / f_msy) if f_msy > 0 else 1.0
    recettes = profit + cout_total

    # Prise en compte sécurisée des coûts nuls (c = 0)
    if cout_total > 0:
        ratio_profit = float(profit / cout_total)
    else:
        ratio_profit = 1.0 if profit > 0 else 0.0

    if recettes > 0:
        marge_nette = float(profit / recettes)
    else:
        marge_nette = -1.0 if profit < 0 else 0.0

    diff_effort = abs(effort_actuel - f_msy)
    
    # 3. Agent Halieutique
    X_test_msy = np.array([[ratio_effort, r_squared]])
    classe_msy = int(clf_msy.predict(X_test_msy)[0])
    
    # 4. Agent Économique
    X_test_mse = np.array([[ratio_profit, marge_nette]])
    classe_mse = int(clf_mse.predict(X_test_mse)[0])
    
    # 5. Agent Décisionnel
    X_test_action = np.array([[ratio_effort, ratio_profit]])
    classe_action = int(clf_action.predict(X_test_action)[0])
    
    # 6. Formatage dynamique des dictionnaires via la méthode .format()
    texte_r2 = DICTIONNAIRE_R2[classe_r2].format(r2=round(r_squared * 100, 2))
    
    texte_msy = DICTIONNAIRE_MSY[classe_msy].format(
        effort=round(effort_actuel), 
        msy=round(f_msy), 
        ratio=round(ratio_effort * 100, 1),
        diff=round(diff_effort)
    )
    
    texte_mse = DICTIONNAIRE_MSE[classe_mse].format(
        profit=f"{profit:,.0f}", 
        abs_profit=f"{abs(profit):,.0f}", 
        marge=round(marge_nette * 100, 1)
    )
    
    texte_action = DICTIONNAIRE_ACTION[classe_action]
    
    # Assemblage final compact
    rapport_final = (
        f"{texte_r2}\n\n"
        f"• {texte_msy}\n"
        f"• {texte_mse}\n"
        f"• {texte_action}"
    )
    
    return rapport_final

import numpy as np
import os
import sys
import time # Pour chronométrer t-SNE

# --- NOUVEAUX IMPORTS ---
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# -------------------------


def load_vectors(absolute_path, file_name):
    """
    Charge des vecteurs à partir d'un fichier texte dans un tableau NumPy.
    
    Chaque ligne est supposée être : ID <espace> vecteur...

    Retours:
        Tuple (numpy.ndarray, numpy.ndarray): (ids, vecteurs)
                                              ou (None, None) si erreur.
    """
    full_path = os.path.join(absolute_path, file_name)
    print(f"Tentative de chargement des vecteurs : {full_path}")

    try:
        vectors = np.loadtxt(full_path)
        print("\nChargement des vecteurs réussi !")
        print(f"Forme du fichier lu (lignes, colonnes) : {vectors.shape}")

        # --- MODIFIÉ : Séparer les IDs des vecteurs ---
        if vectors.ndim == 1:
            # Gérer le cas où il n'y a qu'une seule ligne
            print(f"  -> {vectors.shape[0]} colonnes détectées.")
            ids = vectors[0:1].astype(int) # 1er élément (gardé comme tableau)
            data_vectors = vectors[1:]      # Le reste
        elif vectors.ndim == 2:
            # Cas standard (plusieurs lignes)
            print(f"  -> {vectors.shape[0]} vecteurs chargés.")
            print(f"  -> {vectors.shape[1]} colonnes par vecteur.")
            ids = vectors[:, 0].astype(int) # 1ère colonne
            data_vectors = vectors[:, 1:]   # Toutes les autres colonnes
        
        print(f"  -> Données séparées : {ids.shape[0]} IDs, {data_vectors.shape[1]} dimensions.")
        return ids, data_vectors 

    except FileNotFoundError:
        print(f"\nERREUR : Fichier non trouvé.", file=sys.stderr)
        return None, None
    except ValueError:
        print(f"\nERREUR : Le fichier contient des données non numériques.", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue : {e}", file=sys.stderr)
        return None, None

# --- NOUVEAU : Fonction pour charger les labels ---
def load_labels(absolute_path, file_name):
    """
    Charge les labels (ID, label) à partir d'un fichier texte.
    """
    full_path = os.path.join(absolute_path, file_name)
    print(f"\nTentative de chargement des labels : {full_path}")
    
    try:
        label_data = np.loadtxt(full_path, dtype=int)
        
        print(f"Chargement des labels réussi !")
        print(f"Forme des labels : {label_data.shape}")
        
        # Convertit en dictionnaire (map) pour un accès rapide
        # {1: 'label1', 2: 'label1', 4: 'label2', ...}
        label_map = {node_id: label for node_id, label in label_data}
        print(f"  -> {len(label_map)} labels chargés dans un dictionnaire.")
        return label_map
        
    except FileNotFoundError:
        print(f"\nERREUR : Fichier label non trouvé.", file=sys.stderr)
        return None
    except ValueError:
        print(f"\nERREUR : Le fichier label ne semble pas être au format 'ID Label'.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue : {e}", file=sys.stderr)
        return None
# --- FIN NOUVEAU ---


# --- Exemple d'utilisation ---
if __name__ == "__main__":

    # --- À CONFIGURER ---
    CHEMIN_ABSOLU = r"C:\Users\Maxence\Documents\MVA\Graph Model\ComE\data"
    
    # Fichier des embeddings (ID + vecteur)
    NOM_FICHIER = r"Dblp_alpha-10.0_beta-5.0_ws-10_neg-5_lr-0.025_icom-219_ind-219_k-5_ds-0.0.txt"
    
    # NOUVEAU : Fichier des labels (ID + label)
    NOM_FICHIER_LABELS = r"Dblp\Dblp.labels" 
    # -------------------------

    # --- Chargement des données ---
    vector_ids, data = load_vectors(CHEMIN_ABSOLU, NOM_FICHIER)
    label_map = load_labels(CHEMIN_ABSOLU, NOM_FICHIER_LABELS)

    # Vérification : si les données sont chargées, on peut les utiliser
    if data is not None and label_map is not None:
        print("\n--- Vérification des données chargées ---")
        print(f"Premier ID d'embedding : {vector_ids[0]}")
        print(f"Premier vecteur (10 dims) : {data[0][:10]}")

        # --- NOUVEAU : Alignement des labels ---
        print("\n--- Alignement des labels sur les vecteurs ---")
        
        aligned_labels = []
        missing_nodes = 0
        
        # Boucle sur chaque ID de notre fichier d'embedding
        for vid in vector_ids:
            
            # --- CORRECTION : AJOUT DE +1 ---
            # On cherche le label pour l'ID 'vid + 1' 
            # (ex: l'ID 0 de l'embedding correspond au label 1)
            label = label_map.get(vid + 1, -1) 
            
            if label == -1:
                missing_nodes += 1
                # Affiche l'ID du nœud (index 0) et l'ID recherché (index 1)
                print(f"  -> Nœud sans label trouvé : ID {vid} (recherche sur l'ID {vid + 1})")
            
            aligned_labels.append(label)
        
        # Convertit la liste en tableau NumPy pour matplotlib
        aligned_labels = np.array(aligned_labels)
        
        print(f"Alignement terminé. {len(aligned_labels)} labels alignés.")
        if missing_nodes > 0:
            print(f"ATTENTION : {missing_nodes} vecteurs n'avaient pas de label. Ils seront en couleur -1.")
        # --- FIN NOUVEAU ---


        # --- DÉBUT DE LA VISUALISATION T-SNE ---
        print(f"\n--- Lancement de t-SNE (N={data.shape[0]}) ---")
        start_time = time.time()
        
        tsne = TSNE(n_components=2, 
                    perplexity=30.0, 
                    init='pca', 
                    n_iter=1000, 
                    random_state=42)
        
        tsne_results = tsne.fit_transform(data)
        
        end_time = time.time()
        print(f"t-SNE terminé en {end_time - start_time:.2f} secondes.")

        
        # --- Création du graphique ---
        print("\nAffichage du graphique...")
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(tsne_results[:, 0], 
                              tsne_results[:, 1], 
                              c=aligned_labels,
                              cmap='tab10', 
                              s=10, 
                              alpha=0.7)

        plt.title(f'Visualisation t-SNE des Embeddings (Labels Réels)')
        plt.xlabel('Dimension t-SNE 1')
        plt.ylabel('Dimension t-SNE 2')
        
        # --- AJOUT : Colorbar pour la légende ---
        cbar = plt.colorbar(scatter)
        cbar.set_label('Label Réel (Ground Truth)')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    else:
        print("\nErreur lors du chargement d'un ou plusieurs fichiers. Arrêt du script.", file=sys.stderr)
import numpy as np
from typing import Tuple, List, Dict
from collections import Counter
from copy import deepcopy

class Csp():
    """
        Classe contenant les infos sur les variables à trouver 
        (domaine, compatibilités, contraintes)
    """
    
    def __init__ (self, grille):
        self.grille = grille
        
        
    def initialise_variable(self):
        variable = []
        for i, i_value in enumerate(self.grille.grille):
            for j, j_value in enumerate(i_value):
                if self.grille.grille[i][j] == 0.0 :
                    variable.append((i, j))
        self.variable =  variable
    
    
    def domaine(self, index : Tuple[int, int]) -> List[float] :
    
        return [ i for i in range(1,10) if self.compatible(i, index)]
    
    
    def compatible(self, valeur : float, index : Tuple[int, int]) -> bool :
        v = self.grille.get_valeur_voisins(index)
        if valeur in v[0] or valeur in v[1] or valeur in v[2] :
            return False
        else : 
            return True
    
        
    def initialise_domaine(self):
        csp = {}
        for i, vi in enumerate(self.grille.grille):
            for j, vj in enumerate(vi):
                if len(self.domaine((i, j))) != 0 and self.grille.point_csp_by_index((i, j)) == 0 :
                    csp[(i, j)] = self.domaine((i, j))
        self.domaine_var = csp



class Grille():
    """
        Classe contenant les infos sur la grille à résoudre
    """
    
    def __init__ (self):
        self.grille = np.zeros((9, 9))
    
    def initialise_grille(self, assignement):
        for i in range(9):
            for j in range(9):
                if assignement[(i,j)] != 0 :
                    self.grille[i][j] = assignement[(i,j)]
        
    
    def point_csp_by_index(self, index : Tuple) -> float:
        return self.grille[index[0]][index[1]]
    
    
    
    def get_case(self, index) -> Tuple[List, List]:
        i = index[0] - (index[0] % 3)
        j = index[1] - (index[1] % 3)
        j2 = j
        case = []
        index = []
        for i in range (i, i + 3):
            for j2 in range (j2, j2 + 3):
                #print(i, j2)
                case.append(self.grille[i][j2])
                index.append((i, j2))
            j2 = j
        return case, index
    
    def get_ligne(self, index) -> np.ndarray:
        return self.grille[index[0]]
    
    def get_colone(self, index) -> np.ndarray:
        return self.grille[:,index[1]]
    
    
    def get_valeur_voisins(self, index : Tuple[int, int]) -> Tuple[List] :
        
        ligne = self.get_ligne(index)
        colone = self.get_colone(index)
        case = self.get_case(index)[0]


        return ligne, colone.reshape(colone.shape[0]), np.array(case)
    
    
    def get_index_voisins(self, index) -> List[Tuple[int, int]] :
        index_v = set()
        for i, i_value in enumerate(self.grille):
            for j, j_value in enumerate(i_value):
                if i == index[0] or j == index[1]:
                    index_v.add((i, j))

        
        index_v.update(self.get_case(index)[1])

        
        index_v.remove(index)
        

        index_v = [ i for i in index_v if self.point_csp_by_index(i) == 0.0]

        
        return [ (self.point_csp_by_index((i, j)), (i, j)) for i, j in index_v]
    
    
    def minimum_heuristique_value(self) -> Tuple[int, int]:
        count_contraintes = {}
        for i, i_value in enumerate(self.grille):
            for j, j_value in enumerate(i_value):
                if self.grille[i][j] == 0 and (i, j):
                    v = self.get_valeur_voisins((i,j))
                    v_prim = np.concatenate((v[0], v[1], v[2]), axis = None)
                    n_contraintes = Counter( i for i in v_prim if i != 0 )
                    count_contraintes[(i,j)] = sum(n_contraintes.values())

        count_contraintes = sorted(count_contraintes.items(), key=lambda t: t[1], reverse=True)
        
        return count_contraintes[0][0]



def reviser(voisin : Tuple[int, int], index : Tuple[int, int], csp) -> Tuple[bool, np.ndarray] :
    change = False
    
    for dom in csp.domaine_var[voisin] :
        
        if dom in csp.domaine_var[index] :
            
            csp.domaine_var[voisin].remove(dom)
            change = True
    
    return (change, csp)


def forward_checking(index : Tuple[int, int], grille, csp) -> Tuple :
    
    csp_star = deepcopy(csp)
    for valeur, voisin in grille.get_index_voisins(index) :
        change, csp = reviser(voisin, index, csp)
        if change and len(csp.domaine_var[voisin]) == 0 :
            return (None, False)
        
    return (csp, True)



# ALGORITHME BACKTRACKING SEARCH

def backtrack(assignation : Dict, csp , grille) :
    
    if not 0.0 in assignation.values() : 
        return assignation
    
    else:
        index = grille.minimum_heuristique_value()
        for val in csp.domaine_var[index] :
            
            if csp.compatible(val, index):
                assignation[index] = val
                grille.grille[index[0]][index[1]] = val
                
                csp_star = deepcopy(csp)
                csp_star.domaine_var[index] = [val]
                csp_star, ok = forward_checking(index, grille, csp_star)
                if ok :
                    resultat = backtrack(assignation, csp_star, grille)
                    if resultat is not False : 
                        return resultat
                    
                assignation[index] = 0.0
                grille.grille[index[0]][index[1]] = 0.0
                
                
    return False


def BackTracking_Search(assignation : Dict, csp, grille) :
    
    return backtrack(assignation, csp, grille)

from flask import Flask, request, render_template 


app = Flask(__name__)


@app.route('/', methods =["GET", "POST"])
def gfg():
    dct = {}
    return render_template("index.html", assignement = dct)


@app.route('/Sudoku', methods = ["GET", "POST"])
def Sudoku():
    assignation = {}

    for i in range(9):
        for j in range(9):

            valeur = request.form[f"({i},{j})"]

            valeur = int(valeur) if valeur.isdigit() else valeur

            if valeur in [i for i in range(1, 10)]:
                assignation[(i,j)] = valeur

            else :
                assignation[(i,j)] = 0.0

    grille = Grille()
    grille.initialise_grille(assignation)

    csp = Csp(grille)
    csp.initialise_domaine()
    csp.initialise_variable()

    assignation = BackTracking_Search(assignation, csp, grille)
    print(grille.grille)

    if assignation is not False:
        return render_template("index.html", assignement = assignation, message="Grille résolue")
    else :
        return render_template("index.html", assignement = assignation, message="Grille éronée")

if __name__=='__main__':
   app.run()
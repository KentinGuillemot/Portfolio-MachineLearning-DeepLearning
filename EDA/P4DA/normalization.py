class Normalization():
    def __init__(self, liste):
        self.liste = liste
    
    def methode_min_max(self):
        L = self.liste.copy()

        min_val = min(L)
        max_val = max(L)

        for i in range(len(L)):
            L[i] = round( ( (L[i] - min_val) / (max_val - min_val) ), 3)

        return L
        
    def methode_z_score(self):
        L = self.liste.copy()

        moyenne = sum(L) / len(L)
        variance = sum((x - moyenne) ** 2 for x in L) / (len(L) - 1)  
        ecart_type = variance ** 0.5

        for i in range(len(L)):
            L[i] = round( ( (L[i] - moyenne) / ecart_type), 3 )
        
        return L
    
    def display_list(self):
        return f"My list : {self.liste}"
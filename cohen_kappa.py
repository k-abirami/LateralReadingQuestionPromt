from sklearn.metrics import cohen_kappa_score

ai_rankings = [4, 5, 4, 4, 2, 4, 2, 4, 2, 2]  
my_rankings = [4, 5, 4, 4, 3, 2, 3, 3, 2, 3]  

kappa = cohen_kappa_score(ai_rankings, my_rankings)

print(f"Cohen's Kappa: {kappa:.3f}")
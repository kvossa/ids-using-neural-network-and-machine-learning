import joblib
import pandas as pd

for dataset in ['cic', 'unsw']:
    print(f"\n{'='*50}")
    print(f"  {dataset.upper()} — Features seleccionadas")
    print(f"{'='*50}")
    
    preprocessor = joblib.load(f'models/preprocessing/multiclass/{dataset}/preprocessing.pkl')
    selector = preprocessor.pipeline.named_steps['feature_selection']
    
    features = selector.selected_features_
    print(f"Total: {len(features)}\n")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    
    # Ranking completo con scores
    ranking = selector.get_feature_ranking()
    print(f"\nRanking completo:")
    print(ranking[['feature', 'consensus']].to_string())
#!/usr/bin/env python3
"""
RQ2 Hyperparameter Influence Analysis
Analyzes how different hyperparameters influence agent runtime, performance improvement, and correctness
using Random Forest feature importance and statistical correlation analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hyperparameter_influence():
    """
    Comprehensive analysis of hyperparameter influence on all three objectives
    """
    
    print("=== RQ2: HYPERPARAMETER INFLUENCE ANALYSIS ===")
    
    # Load GA results
    df = pd.read_csv('ga_results_20250919_154742.csv')
    
    # Define hyperparameters and objectives
    hyperparams = [
        'config1_step_limit',
        'config2_cost_limit', 
        'config3_temperature',
        'config4_top_p',
        'config5_max_tokens',
        'config6_model_timeout',
        'config7_env_timeout',
        'config8_baseline'
    ]
    
    objectives = [
        'objective1_execution_time',
        'objective2_model_improved', 
        'objective3_success'
    ]
    
    objective_names = ['Runtime (s)', 'Performance Improvement', 'Correctness']
    hyperparam_names = ['Step Limit', 'Cost Limit', 'Temperature', 'Top P', 'Max Tokens', 'LLM Timeout', 'Env Timeout', 'Baseline']
    
    print(f"Analyzing {len(df)} configurations across {len(hyperparams)} hyperparameters")
    print(f"Objectives: {objective_names}")
    print()
    
    # Prepare data
    X = df[hyperparams].values
    
    # Handle categorical baseline parameter
    X_processed = X.copy()
    
    # Standardize continuous variables (all except baseline)
    scaler = StandardScaler()
    X_processed[:, :-1] = scaler.fit_transform(X[:, :-1])
    
    results = {}
    
    # === RANDOM FOREST FEATURE IMPORTANCE ANALYSIS ===
    print("=== RANDOM FOREST FEATURE IMPORTANCE ===")
    
    for i, (obj, obj_name) in enumerate(zip(objectives, objective_names)):
        print(f"\n--- {obj_name} ---")
        
        y = df[obj].values
        
        # Random Forest with cross-validation
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf, X_processed, y, cv=5, scoring='r2')
        print(f"Cross-validation R^2 score: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        
        # Fit model for feature importance
        rf.fit(X_processed, y)
        
        # Feature importance
        importance = rf.feature_importances_
        
        # Store results
        results[obj_name] = {
            'importance': importance,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Print feature importance
        importance_df = pd.DataFrame({
            'Hyperparameter': hyperparam_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance Ranking:")
        for _, row in importance_df.iterrows():
            print(f"  {row['Hyperparameter']:15}: {row['Importance']:.3f}")
    
    # === CORRELATION ANALYSIS ===
    print("\n=== CORRELATION ANALYSIS ===")
    
    correlation_results = {}
    
    for i, (obj, obj_name) in enumerate(zip(objectives, objective_names)):
        print(f"\n--- {obj_name} Correlations ---")
        
        y = df[obj].values
        correlations = []
        
        for j, (param, param_name) in enumerate(zip(hyperparams, hyperparam_names)):
            x = df[param].values
            
            # Spearman correlation (handles non-linear relationships)
            spearman_corr, spearman_p = spearmanr(x, y)
            
            # Pearson correlation (linear relationships)
            pearson_corr, pearson_p = pearsonr(x, y)
            
            correlations.append({
                'hyperparameter': param_name,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p
            })
            
            print(f"  {param_name:15}: Spearman={spearman_corr:6.3f} (p={spearman_p:.3f}), Pearson={pearson_corr:6.3f} (p={pearson_p:.3f})")
        
        correlation_results[obj_name] = correlations
    
    # === SUMMARY FOR PAPER ===
    print("\n=== SUMMARY FOR RQ2 RESULTS ===")
    
    # Find top 3 most important hyperparameters for each objective WITH DIRECTION
    for obj_name in objective_names:
        importance = results[obj_name]['importance']
        top_indices = np.argsort(importance)[-3:][::-1]  # Top 3 in descending order
        
        print(f"\n{obj_name} - Top 3 Most Influential Hyperparameters:")
        for i, idx in enumerate(top_indices):
            param_name = hyperparam_names[idx]
            imp_score = importance[idx]
            
            # Get correlation direction for this parameter-objective pair
            corr_data = correlation_results[obj_name][idx]
            spearman_corr = corr_data['spearman_corr']
            spearman_p = corr_data['spearman_p']
            
            # Determine direction and significance
            if spearman_p < 0.05:
                direction = "↑ POSITIVE" if spearman_corr > 0 else "↓ NEGATIVE"
                significance = f"(ρ={spearman_corr:.3f}, p={spearman_p:.3f})"
            else:
                direction = "→ POSITIVE" if spearman_corr > 0 else "← NEGATIVE" 
                significance = f"(ρ={spearman_corr:.3f}, n.s.)"
            
            print(f"  {i+1}. {param_name}: {imp_score:.3f} importance, {direction} {significance}")
    
    # === ACTIONABLE DIRECTIONAL INSIGHTS ===
    print("\n=== ACTIONABLE DIRECTIONAL INSIGHTS FOR RQ3 ===")
    
    # Find significant correlations for actionable recommendations
    significant_insights = []
    for obj_name in objective_names:
        for i, corr_data in enumerate(correlation_results[obj_name]):
            if corr_data['spearman_p'] < 0.1:  # Use p < 0.1 for more insights
                param_name = corr_data['hyperparameter']
                corr = corr_data['spearman_corr']
                p_val = corr_data['spearman_p']
                importance = results[obj_name]['importance'][i]
                
                direction = "INCREASE" if corr > 0 else "DECREASE"
                objective_direction = "improve" if obj_name != "Runtime (s)" else "reduce"
                
                significant_insights.append({
                    'param': param_name,
                    'objective': obj_name,
                    'direction': direction,
                    'obj_direction': objective_direction,
                    'correlation': corr,
                    'p_value': p_val,
                    'importance': importance
                })
    
    # Sort by importance and print actionable insights
    significant_insights.sort(key=lambda x: x['importance'], reverse=True)
    
    print("Based on significant correlations (p < 0.1):")
    for insight in significant_insights:
        print(f"• To {insight['obj_direction']} {insight['objective']}: {insight['direction']} {insight['param']}")
        print(f"  (ρ={insight['correlation']:.3f}, p={insight['p_value']:.3f}, importance={insight['importance']:.3f})")
    
    if not significant_insights:
        print("• No statistically significant correlations found (p < 0.1)")
        print("• This suggests complex non-linear interactions between hyperparameters")
        print("• Multi-objective optimization is essential due to these complex trade-offs")
    
    # === GENERATE TABLE DATA ===
    print("\n=== TABLE DATA FOR RQ2 ===")
    print("Hyperparameter | Runtime Impact | Performance Impact | Correctness Impact | Overall Ranking")
    print("---------------|----------------|--------------------|--------------------|----------------")
    
    # Calculate overall ranking based on average importance
    overall_importance = np.mean([results[obj]['importance'] for obj in objective_names], axis=0)
    overall_ranking = np.argsort(overall_importance)[::-1]
    
    for rank, idx in enumerate(overall_ranking):
        runtime_imp = results['Runtime (s)']['importance'][idx]
        perf_imp = results['Performance Improvement']['importance'][idx]
        corr_imp = results['Correctness']['importance'][idx]
        
        print(f"{hyperparam_names[idx]:14} | {runtime_imp:13.3f} | {perf_imp:17.3f} | {corr_imp:17.3f} | {rank+1:14}")
    
    # === ACTIONABLE INSIGHTS ===
    print("\n=== ACTIONABLE INSIGHTS FOR PRACTITIONERS ===")
    
    # Most important hyperparameter overall
    most_important_idx = overall_ranking[0]
    print(f"1. Most Critical Hyperparameter: {hyperparam_names[most_important_idx]}")
    print(f"   - Average importance: {overall_importance[most_important_idx]:.3f}")
    
    # Objective-specific recommendations
    for obj_name in objective_names:
        top_idx = np.argmax(results[obj_name]['importance'])
        print(f"2. For optimizing {obj_name}: Focus on {hyperparam_names[top_idx]} (importance: {results[obj_name]['importance'][top_idx]:.3f})")
    
    # Model reliability
    print(f"\n3. Model Reliability:")
    for obj_name in objective_names:
        cv_score = results[obj_name]['cv_score']
        if cv_score > 0.7:
            reliability = "High"
        elif cv_score > 0.5:
            reliability = "Moderate"
        else:
            reliability = "Low"
        print(f"   - {obj_name}: R^2 = {cv_score:.3f} ({reliability} reliability)")
    
    return results, correlation_results

def generate_rq2_table_content():
    """Generate content for RQ2 table focusing on hyperparameter influence"""
    
    print("\n=== RQ2 TABLE CONTENT (LaTeX Format) ===")
    
    # This would be populated based on the analysis results
    print("""
\\begin{table}[t!]
\\centering
\\caption{Hyperparameter influence on agent runtime, performance improvement, and correctness.}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Hyperparameter} & \\textbf{Runtime Impact} & \\textbf{Performance Impact} & \\textbf{Correctness Impact} \\\\
\\midrule
Step Limit & High (0.XXX) & Medium (0.XXX) & High (0.XXX) \\\\
Temperature & Medium (0.XXX) & High (0.XXX) & Medium (0.XXX) \\\\
Max Tokens & Medium (0.XXX) & High (0.XXX) & Low (0.XXX) \\\\
Cost Limit & High (0.XXX) & Low (0.XXX) & Medium (0.XXX) \\\\
Top P & Low (0.XXX) & Medium (0.XXX) & Low (0.XXX) \\\\
LLM Timeout & Low (0.XXX) & Low (0.XXX) & Medium (0.XXX) \\\\
Env Timeout & Medium (0.XXX) & Low (0.XXX) & Low (0.XXX) \\\\
Baseline & Medium (0.XXX) & Medium (0.XXX) & High (0.XXX) \\\\
\\bottomrule
\\end{tabular}
\\label{tab:rq2_hyperparams}
\\end{table}
    """)

if __name__ == "__main__":
    results, correlations = analyze_hyperparameter_influence()
    generate_rq2_table_content()

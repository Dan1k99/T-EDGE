import os
from tGraphNE import tGraphNE
from tGraph import tGraph
from link_prediction import LP

def main():
    datasets = ["Eth_0_999k", "Eth_1M_1_99M"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filetype = "pkl_f"
    
    # Parameter setting
    dimensions = 128
    window_size = 4
    workers = 8
    num_walks = 10
    walk_length = 10
    
    # Generate Temporal Walks (WS1 + WS5): time_close_exp and amount_exp with alpha=0.5
    first_biased_type = "time_uniform"
    time_biased_type = "time_close_exp"
    amount_biased = "amount_exp"
    alpha = 0.5
    edge_score_mode = "append"
    model_type = "XGBoost"
    gamma = 0.2
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"Executing A/B Testing Pipeline for Dataset: {dataset}")
        print(f"{'#'*60}")
        
        file_ = os.path.join(script_dir, dataset + "_df_train_0.5.pickle")
        if not os.path.exists(file_):
            print(f"Dataset {file_} not found. Please run prepare_dataset.py first.")
            continue
            
        print(f"Loading graph A (including failed transactions) from {file_}...")
        tG_A = tGraph(file_, filetype, exclude_failed=False)
        print(tG_A)
        
        print(f"Loading graph B (excluding failed transactions) from {file_}...")
        tG_B = tGraph(file_, filetype, exclude_failed=True)
        print(tG_B)
        
        out_dir = os.path.join(script_dir, "..", "data", dataset)
        os.makedirs(out_dir, exist_ok=True)
        
        # Run A (Baseline)
        print("\n========== RUN A: Baseline (XGBoost) ==========")
        output_A = os.path.join(out_dir, f"vec_baseline_{dataset}.txt")
        print("Starting baseline TWMDG embeddings generation (WS1 + WS5)...")
        tGraphNE(tG_A, time_biased_type, first_biased_type, amount_biased, alpha, dimensions, num_walks, walk_length, output_A)
        print("Starting Link Prediction Pipeline (Baseline)...")
        roc_A, ap_A = LP(dataset, output_A, edge_score_mode, tG=tG_A, model_type=model_type, use_gas_features=False)
        
        # Run B (Embedding Without Failed Transactions)
        print("\n========== RUN B: Embedding Without Failed Transactions (XGBoost) ==========")
        output_B = os.path.join(out_dir, f"vec_no_failed_{dataset}.txt")
        print("Starting TWMDG embeddings generation (excluding failed transactions)...")
        tGraphNE(tG_B, time_biased_type, first_biased_type, amount_biased, alpha, dimensions, num_walks, walk_length, output_B)
        print("Starting Link Prediction Pipeline (No Failed Transactions)...")
        roc_B, ap_B = LP(dataset, output_B, edge_score_mode, tG=tG_B, model_type=model_type, use_gas_features=False)
        
        # Comparative Report
        print(f"\n{'='*50}")
        print(f"FINAL COMPARATIVE REPORT: {dataset}")
        print(f"{'='*50}")
        print(f"Metric\t\tBaseline\tWithout Failed Tx")
        print(f"ROC-AUC\t\t{roc_A:.4f}\t\t{roc_B:.4f}")
        print(f"Avg Precision\t{ap_A:.4f}\t\t{ap_B:.4f}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

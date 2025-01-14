# bin/bash
data_repo="experiments/data/data_refined_final"
output_file="results_spacy_refined_rerank.json"
log_file="experiments/logs/results_spacy_refined_rerank.log"
method="spacy"
rerank=True

python test_retrieval.py\
    --data-repo $data_repo\
    --output-file $output_file\
    --method $method\
    --rerank $rerank >> $log_file
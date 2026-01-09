```text
journal-landscapes-attention/
│
├── [0]get_articles.py                # Select articles from OpenAlex dump based on journal list
├── [1]data_cleaning.ipynb           # Initial data cleaning and preprocessing
│
├── [2]labeling.ipynb                # GPT labeling: identify aging-related articles
├── [3]classification.ipynb          # Train aging classifier and apply to full dataset
├── [4]classification2.ipynb         # Sentence embedding of non-aging articles
├── [5]classification3.ipynb         # Unsupervised clustering of non-aging articles
├── [6]classification4.ipynb         # GPT-4.1 naming of unsupervised clusters
│
├── [7]building_network.py           # Extract citation edge list
├── [8]build_network.ipynb           # Build networks and compute network metrics
│
├── [9]main.ipynb                    # Main analyses and selected appendix results
├── [10]surv.ipynb                   # Construct data for Cox models
├── [11]survival_analysis.nb.html    # Cox models and robustness checks (R output)
│
├── [Appendix]alternative_classifications.ipynb  # Alternative LLM-based classifications
├── [Appendix]get_other_fields.py                # Retrieve other fields from OpenAlex for dispersion reference
└── [Appendix]gpt_oss.ipynb                      # Open-weight LLM reproducibility check

```

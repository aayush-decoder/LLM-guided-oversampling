import json

nb_path = 'd:/project/College/ML-project/my_approch/new_approch.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "id": "eval_md_1",
        "metadata": {},
        "source": [
            "## 9. Final Generation & Dataset Augmentation\n",
            "Now that the LLM is fine-tuned to generate valid, realistic minority class samples, we will use it to generate a final batch of synthetic data and combine it with the original training set."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "eval_code_1",
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Generating final synthetic minority class samples with fine-tuned LLM...')\n",
            "\n",
            "FINAL_CANDIDATES = 100\n",
            "final_raw_candidates = generate_candidates(model, tokenizer, num_candidates=FINAL_CANDIDATES, temperature=0.7)\n",
            "\n",
            "final_valid_points = []\n",
            "for prompt_str, resp_text in final_raw_candidates:\n",
            "    parsed_dict = deserialize_response(resp_text)\n",
            "    is_valid, _ = evaluate_candidate(parsed_dict)\n",
            "    if is_valid:\n",
            "        final_valid_points.append(parsed_dict)\n",
            "\n",
            "print(f'Final Generator Yield: {len(final_valid_points)}/{FINAL_CANDIDATES} ({(len(final_valid_points)/FINAL_CANDIDATES)*100:.1f}%)')\n",
            "\n",
            "# Include accumulated valid data from earlier rounds as well\n",
            "all_synthetic_dicts = accumulated_synthetic_data + final_valid_points\n",
            "synthetic_df = pd.DataFrame(all_synthetic_dicts)[feature_names]\n",
            "\n",
            "# Combine with original training data\n",
            "X_train_aug = pd.concat([X_train, synthetic_df], ignore_index=True)\n",
            "y_train_aug = pd.concat([y_train, pd.Series([1]*len(synthetic_df))], ignore_index=True)\n",
            "\n",
            "print(f'Original Training Data Shape: {X_train.shape}')\n",
            "print(f'Augmented Training Data Shape: {X_train_aug.shape}')\n",
            "print(f'Original Class Ratio: {y_train.value_counts(normalize=True).to_dict()}')\n",
            "print(f'Augmented Class Ratio: {y_train_aug.value_counts(normalize=True).to_dict()}')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "eval_md_2",
        "metadata": {},
        "source": [
            "## 10. Evaluation: Baseline vs. Augmented Model\n",
            "We train a new Random Forest on the augmented dataset and compare its performance (Precision, Recall, F1, ROC-AUC) on the untouched test set."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "eval_code_2",
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n",
            "\n",
            "# Train model on Augmented Data\n",
            "aug_rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
            "aug_rf.fit(X_train_aug, y_train_aug)\n",
            "\n",
            "# Predictions\n",
            "base_preds = baseline_rf.predict(X_test)\n",
            "base_probs = baseline_rf.predict_proba(X_test)[:, 1]\n",
            "\n",
            "aug_preds = aug_rf.predict(X_test)\n",
            "aug_probs = aug_rf.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print('=== Baseline Model (Trained on Original Imbalanced Data) ===')\n",
            "print(classification_report(y_test, base_preds))\n",
            "print(f'ROC AUC Score: {roc_auc_score(y_test, base_probs):.4f}\\n')\n",
            "\n",
            "print('=== Augmented Model (Trained on LLM-Synthesized Balanced Data) ===')\n",
            "print(classification_report(y_test, aug_preds))\n",
            "print(f'ROC AUC Score: {roc_auc_score(y_test, aug_probs):.4f}\\n')\n",
            "\n",
            "print('Confusion Matrix (Baseline):')\n",
            "print(confusion_matrix(y_test, base_preds))\n",
            "print('\\nConfusion Matrix (Augmented):')\n",
            "print(confusion_matrix(y_test, aug_preds))\n"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Evaluation cells appended successfully.')

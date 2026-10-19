# v3 smoke report  (dataset=iris, llm=tiny-gpt2)

- PASS **original**: accuracy=0.9333 (original 0.9333) audit=WARN
- PASS **pred_llm_v3**: accuracy=0.6333 (original 0.9333) audit=PASS
- PASS **v3_no_negative**: accuracy=0.6333 (original 0.9333) audit=PASS
- PASS **v3_no_outlier_range**: accuracy=0.5 (original 0.9333) audit=PASS

Artifacts:
- accuracy: D:\project\College\ML-project\v3\runs\_smoke\tables\table_accuracy.md
- f1_macro: D:\project\College\ML-project\v3\runs\_smoke\tables\table_f1_macro.md
- fidelity: D:\project\College\ML-project\v3\runs\_smoke\tables\fidelity.md

OVERALL: PASS
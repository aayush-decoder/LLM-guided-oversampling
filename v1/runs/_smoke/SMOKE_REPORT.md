# v1 smoke report  (dataset=iris, llm=distilgpt2)

- PASS **original**: accuracy=0.9333 (original 0.9333) audit=WARN
- PASS **pred_llm**: accuracy=1.0 (original 0.9333) audit=WARN
- PASS **great**: accuracy=0.5333 (original 0.9333) audit=PASS
- PASS **taptap**: accuracy=0.9333 (original 0.9333) audit=WARN
- PASS **ctgan**: accuracy=0.4333 (original 0.9333) audit=PASS

Artifacts:
- accuracy: D:\project\College\ML-project\v1\runs\_smoke\tables\table_accuracy.md
- f1_macro: D:\project\College\ML-project\v1\runs\_smoke\tables\table_f1_macro.md
- fidelity: D:\project\College\ML-project\v1\runs\_smoke\tables\fidelity.md

OVERALL: PASS
# v2 smoke report  (dataset=iris, llm=tiny-gpt2)

- PASS **original**: accuracy=0.9333 (original 0.9333) audit=WARN
- PASS **pred_llm_v2**: accuracy=0.4 (original 0.9333) audit=PASS
- PASS **v2_encoded**: accuracy=0.5333 (original 0.9333) audit=PASS
- PASS **v2_plain_loss**: accuracy=0.4 (original 0.9333) audit=PASS

Artifacts:
- accuracy: D:\project\College\ML-project\v2\runs\_smoke\tables\table_accuracy.md
- fidelity: D:\project\College\ML-project\v2\runs\_smoke\tables\fidelity.md

OVERALL: PASS
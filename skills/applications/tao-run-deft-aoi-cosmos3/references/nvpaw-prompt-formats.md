# NVPAW task and prompt formats

| Task type | Image roles | Evaluator family |
| --- | --- | --- |
| Component Count | target | counting |
| Component Classification | target | classification |
| Component Detection | target | detection |
| Defect Classification | target | classification |
| Defect Detection | target | detection |
| Ref_based Defect Classification | golden, target | classification |
| Ref_based Defect Detection | golden, target | detection |

Prompts and ground truth remain in their native message content. Classification
may use direct defect-presence text, BCQ, or MCQ option responses. Detection
answers use integer `[x1,y1,x2,y2]` coordinates normalized to `[0,1000]`.
Component Count answers remain the official integer-count response from the
Mining annotation; they are training-only when the frozen KPI split omits that
task.
The app does not paraphrase prompts, reorder images, or shorten generation for
detection. The workspace evaluator is the parsing and scoring authority.

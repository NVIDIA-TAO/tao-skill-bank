# Platform nightly — template contract

- **SKIP** `k8s/action-job.yaml.tmpl` — neither kubeconform nor kubectl installed
- **SKIP** `k8s/indexed-job.yaml.tmpl` — neither kubeconform nor kubectl installed
- **SKIP** `k8s/single-pod-job.yaml.tmpl` — neither kubeconform nor kubectl installed
- **PASS** `slurm/multinode.sbatch.tmpl` — bash -n clean (sbatch absent — directives unchecked)
- **PASS** `slurm/singlenode.sbatch.tmpl` — bash -n clean (sbatch absent — directives unchecked)

5 checked, 0 failed.

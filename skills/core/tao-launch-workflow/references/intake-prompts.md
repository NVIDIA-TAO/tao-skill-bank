# Missing-input prompt shape

When asking for launch inputs, include concrete examples and both dataset input
modes. Do not ask only for "dataset root".

Use this structure and adapt spec keys to the selected model/action:

```text
I need these launch inputs before I can create specs or runner files:

1. Execution platform: brev, slurm, local-docker, or kubernetes.

2. Dataset inputs. You can provide either mode:
   A) Root mode: give train/eval roots and I map required files automatically.
      Example Cosmos-RL:
      train_root=/lustre/fsw/.../cosmos/train
      -> custom.train_dataset.annotation_path=train_root/annotations.json
      -> custom.train_dataset.media_path=train_root
   B) Direct spec mode: give the exact config/spec parameters yourself.
      Example:
      custom.train_dataset.annotation_path=/lustre/fsw/.../train_annotations.json
      custom.train_dataset.media_path=/lustre/fsw/.../videos_train.tar.gz
      custom.val_dataset.annotation_path=/lustre/fsw/.../eval_annotations.json
      custom.val_dataset.media_path=/lustre/fsw/.../eval_videos/

   Platform examples:
   - SLURM/Lustre: /lustre/fsw/.../data/train or lustre:///lustre/fsw/.../data/train
   - Brev/Kubernetes: s3://bucket/path/train and s3://bucket/path/eval
   - local-docker: /data/tao/<model>/train or file:///data/tao/<model>/eval

3. Container image. I will resolve the default from packaged model metadata and
   show it before launch, for example:
   default image for <model>/<action>: <resolved container image>
   Use this image, or provide image=<override> to pin a different TAO build.

4. Compute shape required by the model, for example GPUs/nodes.

5. Required credentials from platform/model docs, for example HF_TOKEN for
   gated Hugging Face models.

6. Monitoring preference. By default I monitor in this chat and post progress
   every 5 minutes; choose 1-2 minutes for smoke tests or 10-15 minutes for
   long training.
```

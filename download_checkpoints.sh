#!/bin/sh

root="$(dirname "$0")"
checkpoints_path="$root/checkpoints"

gsutil -m cp gs://morphic-research-assets/inspyrenet-base.pth "$checkpoints_path"
gsutil -m cp gs://morphic-research-assets/inspyrenet-fast.pth "$checkpoints_path"
gsutil -m cp gs://morphic-research-assets/inspyrenet-nightly.pth "$checkpoints_path"
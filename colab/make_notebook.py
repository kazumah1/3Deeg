"""Regenerate colab/3deeg_pipeline.ipynb from the cell list below (keeps the notebook diff-able)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 3Deeg pipeline on Colab\n"
       "Code lives in the repo (`n3d/`); this notebook only runs it. Everything persistent "
       "(data, checkpoints, the compiled PVCNN ops) goes to Drive under `MyDrive/3deeg/`, so a "
       "disconnect costs at most the steps since the last checkpoint. Re-run cell 1 after every new runtime.\n\n"
       "Order: 1 setup, 2 data (once), 3 encoder overfit check, 4 full encoder, 5 diagnostics and baselines, "
       "6 PVCNN build check, 7 diffusion overfit check, 8 full diffusion (expensive)."),
    md("## 1. Setup (every new runtime)"),
    code("""from google.colab import drive
drive.mount('/content/drive')
import os, subprocess
BRANCH = 'colab-pipeline'
os.environ.update({
    'N3D_DATA': '/content/drive/MyDrive/3deeg/data/neuro-3d',
    'N3D_RUNS': '/content/drive/MyDrive/3deeg/runs',
    'N3D_UPSTREAM': '/content/n3d_cache',
    'TORCH_EXTENSIONS_DIR': '/content/drive/MyDrive/3deeg/torch_extensions',
})
import torch
if not torch.cuda.is_available():
    print('WARNING: CPU-only runtime. Runtime > Change runtime type > GPU, then re-run this cell. '
          'Encoder training is ~50x slower on CPU and PVCNN cannot run at all.')
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
%cd /content
!rm -rf /content/3deeg && git clone -q -b $BRANCH https://github.com/kazumah1/3deeg /content/3deeg
%cd /content/3deeg
!git log --oneline -1
!pip -q install -r requirements-colab.txt
from n3d import viz"""),
    md("## 2. Data (once; ~1.1 GB for sub01, kept on Drive)\n"
       "Add participants later with `--participants sub02 sub03 ...`; already-packed files are skipped."),
    code("!python -m n3d.download --participants sub01"),
    md("## 3. Encoder overfit check (4 categories, ~1-2 min on GPU)\n"
       "The pass condition is `trainset_shape_top1` climbing toward 1.0 (the model can memorise its own training "
       "objects). Val has only 4 objects here, so its accuracy moves in steps of 0.25 and is not the check."),
    code("""!python -m n3d.encoder train --run enc_tiny --tiny --epochs 60 --batch-size 32 --restart
viz.curves('enc_tiny', ['train_loss', 'trainset_shape_top1', 'val_shape_top1']).show()"""),
    md("## 4. Full encoder, sub01 (upstream recipe: 200 epochs)\n"
       "Resumable: re-run the same cell after a disconnect. Selection is on val (variant 07); test is only reported."),
    code("""!python -m n3d.encoder train --run enc_sub01 --epochs 200
viz.curves('enc_sub01', ['val_shape_nll']).show()
viz.curves('enc_sub01', ['trainset_shape_top1', 'val_shape_top5', 'test_shape_top5', 'val_shape_top1', 'test_shape_top1']).show()"""),
    md("## 5. Failure-point diagnostics and reference baselines (cheap)\n"
       "* `diagnose`: averaged vs single-trial EEG, each stream zeroed, shuffled EEG, noise sweep, within-category 2AFC, "
       "within-category shape probe.\n"
       "* `baselines`: random / same-category / category-medoid / oracle-nearest Chamfer, plus classify-then-template "
       "and retrieve-then-copy from the encoder. Any generator must beat classify-then-template to claim EEG-driven shape."),
    code("""!python -m n3d.encoder diagnose --run enc_sub01
!python -m n3d.encoder embed --run enc_sub01
!python -m n3d.baselines --run baselines_sub01 --encoder-run enc_sub01"""),
    md("## 6. PVCNN build check (GPU; first time 3-8 min, cached on Drive afterwards)"),
    code("""!python -c "from n3d.upstream import pvcnn2_class; pvcnn2_class(); print('PVCNN ops OK')"
"""),
    md("## 7. Diffusion overfit check (8 training objects)\n"
       "Generated clouds should converge to the training shapes; val_cd here is on those same objects."),
    code("""!python -m n3d.diffusion train --run dm_tiny --encoder-run enc_sub01 --tiny --max-steps 3000 --batch-size 8 --eval-every 500 --n-eval 8 --restart
viz.curves('dm_tiny', ['loss', 'val_cd']).show()
viz.show_samples('dm_tiny', k=3)"""),
    md("## 8. Full diffusion, sub01 (expensive)\n"
       "Upstream trains ~100k steps at batch 48. Check `sec_per_step` in the log after a few hundred steps and "
       "estimate cost before letting it run. Lower `--batch-size` if the GPU runs out of memory."),
    code("""!python -m n3d.diffusion train --run dm_sub01 --encoder-run enc_sub01 --max-steps 100000"""),
    md("### Evaluate and swap test"),
    code("""!python -m n3d.diffusion sample --run dm_sub01 --eval-split test --samples 5 --steps 100
!python -m n3d.diffusion swap --run dm_sub01 --eval-split test --steps 100
viz.show_samples('dm_sub01', file='pred_test_k5_s100.npz', k=4)"""),
    md("### Cheap comparison runs (PointNet denoiser; Branch A/B)\n"
       "Same data and harness; swap one factor at a time."),
    code("""# conditioning: concat vs FiLM, same denoiser
!python -m n3d.diffusion train --run pn_concat --encoder-run enc_sub01 --denoiser pointnet --cond-mode concat --max-steps 30000
!python -m n3d.diffusion train --run pn_film --encoder-run enc_sub01 --denoiser pointnet --cond-mode film --max-steps 30000
# what the decoder could do with a perfect encoder, and with category alone
!python -m n3d.diffusion train --run pn_oracle_clip --cond-source clip --denoiser pointnet --max-steps 30000
!python -m n3d.diffusion train --run pn_category --cond-source category --denoiser pointnet --max-steps 30000"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": []},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("3deeg_pipeline.ipynb")
nbf.write(nb, out)
print(out)

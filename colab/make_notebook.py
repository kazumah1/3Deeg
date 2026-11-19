"""Regenerate colab/3deeg_pipeline.ipynb from the cell list below (keeps the notebook diff-able)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 3Deeg pipeline on Colab\n"
       "Packed data comes from the private Hugging Face cache into runtime disk. "
       "Drive holds checkpoints and compiled ops only. Re-run setup and data cells after a new runtime. "
       "Run cells 1–2 for a CPU loading check; later cells start training."),
    md("## 1. Setup (every new runtime)\n"
       "Add a read-only Hugging Face token as Colab secret `HF_TOKEN` and allow this notebook access. "
       "Your Hugging Face account must have access to `kazumah/3deeg-packed`. Never paste tokens into cells."),
    code("""import os, subprocess, sys
from pathlib import Path
from google.colab import drive, userdata
from huggingface_hub import login
drive.mount('/content/drive')  # checkpoints only; packed data does not use Drive
login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)
BRANCH = 'codex/hf-packed-data'
os.environ.update({
    'N3D_DATA': '/content/3deeg-data',
    'N3D_RUNS': '/content/drive/MyDrive/3deeg/runs',
    'N3D_UPSTREAM': '/content/n3d_cache',
    'TORCH_EXTENSIONS_DIR': '/content/drive/MyDrive/3deeg/torch_extensions',
})
repo = Path('/content/3deeg-hf')
if not repo.exists():
    subprocess.run(['git', 'clone', '-q', '-b', BRANCH,
                    'https://github.com/kazumah1/3Deeg.git', str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
from n3d import viz
print('Data:', os.environ['N3D_DATA'])"""),
    md("## 2. Fetch and verify packed data (CPU; every new runtime)\n"
       "About 521 MB for one subject and shared arrays; all 12 subjects plus both geometries total 4.80 GB. "
       "The downloader pins the dataset revision, checks every selected file's SHA-256, validates arrays, "
       "and loads a real CPU dataset item. It does not train. Add subjects to `PARTICIPANTS` as needed."),
    code("""from n3d.hub import fetch, validate_arrays, smoke_loader
PARTICIPANTS = ['sub01']
root = fetch(PARTICIPANTS, data_root=os.environ['N3D_DATA'])
print(validate_arrays(root, PARTICIPANTS))
for subject in PARTICIPANTS:
    print(smoke_loader(root, subject))"""),
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
nb = nbf.v4.new_notebook(cells=cells, metadata={"colab": {"provenance": []},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("3deeg_pipeline.ipynb")
nbf.write(nb, out)
print(out)

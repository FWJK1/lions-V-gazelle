#!/bin/bash
#SBATCH --job-name=lions_gazelle
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fkeenank@uvm.edu
#SBATCH --output=/gpfs1/home/f/k/fkeenank/logs/lions-slurm-%j.out
#SBATCH --error=/gpfs1/home/f/k/fkeenank/logs/lions-slurm-%j.err

set -e
module purge
module load python3.12-anaconda/2024.06-1
source activate data-science

my_job_header
echo "Python: $(python --version) | Env: $CONDA_DEFAULT_ENV | Start: $(date)"

cd /gpfs1/home/f/k/fkeenank/lions-V-gazelle
python src/main.py --title "zero-vec-update"

echo "End: $(date)"
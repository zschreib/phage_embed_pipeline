# Phage Protein Embedding Framework

A codebase for building and exploring latent spaces of phage replication proteins using ESM-based embeddings, clustering, and gene neighborhood analysis. Documentation and pipeline are under active development.

### Dependencies

* `phage_embed_pipeline` can be ran either on a machine with a properly set up python environment or by creating a custom [Conda](https://docs.conda.io/en/latest/) environment if you do not wish to change your current setup (recommended).

### Installation

* Conda env setup (recommended) [Help](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```
#Retrieve repo
git clone https://github.com/zschreib/phage_embed_pipeline.git
cd phage_embed_pipeline

#Create environment
conda env create -n phage_embed python=3.11 -y

#Activate environment
conda activate phage_embed

#Setup env and install dependencies
pip install -e .

```

{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 ArialMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww13440\viewh10200\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs28 \cf0 # CMC: Canonical Microcircuit Neural ODE Vision Models\
\
This project implements biologically inspired neural ODE models based on **canonical microcircuits (CMC)** for visual processing tasks. The architecture includes a retinal preprocessing layer and a hierarchical V1\'96V5 visual cortex model. Models are trainable end-to-end on standard vision benchmarks.\
\
## \uc0\u10024  Features\
\
- **Flexible CMC Modules** \'97 Layer-specific, parameterized cortical dynamics\
- **Retinal Processing Layer** \'97 ON/OFF pathways, center-surround fields\
- **Visual Hierarchy (V1\'96V5)** \'97 Laminar and recurrent processing\
- **Multi-Dataset Support** \'97 MNIST, CIFAR-10/100, Tiny ImageNet\
- **Analysis Tools** \'97 Phase space, attractor dynamics, loss landscapes\
\
## \uc0\u55357 \u56615  Installation\
\
```bash\
# Clone the repository\
git clone https://github.com/yourusername/CMC.git\
cd CMC\
\
# Install dependencies\
pip install -r requirements.txt\
# Or manually:\
pip install torch torchvision torchdiffeq numpy matplotlib seaborn scikit-learn plotly scipy networkx tqdm\
\
\pard\pardeftab720\partightenfactor0

\fs26 \cf2 ## Project Structure\
\
```\
project_root/\
\uc0\u9500 \u9472 \u9472  models/\
\uc0\u9474    \u9500 \u9472 \u9472  canonical_circuit.py       # Base CMC model\
\uc0\u9474    \u9500 \u9472 \u9472  flexible_CMC.py             # Flexible multi-node CMC\
\uc0\u9474    \u9500 \u9472 \u9472  flexible_CMC_retina.py    # CMC with retinal preprocessing\
\uc0\u9474    \u9474 \
\uc0\u9500 \u9472 \u9472  tasks/\
\uc0\u9474    \u9492 \u9472 \u9472  multi_dataset_tasks.py    # Training/evaluation for multiple datasets\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  visualization/\
\uc0\u9474    \u9500 \u9472 \u9472  attractor_analyzer.py     # Attractor analysis tools\
\uc0\u9474    \u9500 \u9472 \u9472  dynamics_visualizer.py    # Dynamics visualization\
\uc0\u9474    \u9492 \u9472 \u9472  phase_diagrams.py         # Phase space diagrams\
\uc0\u9474    \
	\uc0\u9492 \u9472 \u9472  loss surface.py         # Loss surfaces\
\
\uc0\u9500 \u9472 \u9472  experiments/\
\uc0\u9474    \u9492 \u9472 \u9472  experiment_runner.py      # Experiment management\
\uc0\u9474 \
\uc0\u9492 \u9472 \u9472  comprehensive_experiments.py          # Main entry point\
\
\
\
\pard\pardeftab720\partightenfactor0
\cf2 ## Usage\
\
### Basic Training\
\
Train a retinal CMC model on MNIST:\
```bash\
python comprehensive_experiments.py --experiment train --dataset mnist --model_type retinal_cmc --epochs 10\
```\
Default will run on 1-5 CMC neural ODE nodes\
}
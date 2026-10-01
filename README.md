# Speech Processing Methods & Technologies

Lab assignments for the "Speech Processing Methods & Technologies" course (Master's program).

## Semester 2

C++11 and Python 3, standard library only. Run `make test` inside a lab directory.

| Lab | Topic |
|-----|--------|
| [Sem 2/Lab1](Sem%202/Lab1/) | Discretization |
| [Sem 2/Lab2](Sem%202/Lab2/) | K-Means |
| [Sem 2/Lab3](Sem%202/Lab3/) | Hidden Markov Model |
| [Sem 2/Lab4](Sem%202/Lab4/) | Multilayer perceptron |

## Semester 3

PyTorch, conda environment `base`. Reports are LaTeX sources compiled with `xelatex report.tex`.

| Lab | Topic |
|-----|--------|
| [Sem3/Lab1](Sem3/Lab1/) | Convolutional networks on CIFAR-10 |
| [Sem3/Lab2](Sem3/Lab2/) | Restricted Boltzmann machines, 8× image compression |

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd Sem3/Lab1 && python train.py && xelatex report.tex
cd ../Lab2 && python train.py && xelatex report.tex
```

CIFAR-10 is downloaded into `Sem3/data/` on the first run and is not stored in git.

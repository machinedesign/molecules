Molecule generation using deep learning and evaluation tools from rdkit.

![molecules](http://i.imgur.com/WmQaZRK.png)

Download and preprocess data steps
==================================

The following datasets are provided : chembl22, zinc12 and zinc15.
The first step is to download them using scripts
provided in data/raw/chembl22, data/raw/zinc12 and data/raw/zinc15.

For zinc12 (the same for zinc15 and chembl22, just replace zinc12 by zinc15 or chembl22):

```bash
cd data/raw/zinc12
bash get.sh
```

This will create a file **data/raw/zinc12.csv**

The second step is to preprocess the data.
The goal of preprocessing is to convert data to numpy format (but still strings),
shuffle the data, and select only strings which dont exceed some maximum length.
the default maximum length is 120.

```bash
python tools/preprocess.py data/raw/zinc12.csv data/zinc12.npz
```
This will create a file in **data/zinc12.npz**

Run example
===========

Once data/zinc12.npz has been created, the example can be run:

```bash
cd examples/
python example.py
```

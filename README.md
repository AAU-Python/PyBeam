# PyBeam

This is a Python package for working with 2D Bernoulli-Euler beam element models.

## Getting started

Please make sure that you've followed the setup guide at <https://github.com/AAU-Python>.

### Obtain a copy of this repository

To obtain a copy of this repository, open a terminal (for example PowerShell). Then, run the following commands:

```shell
cd <directory where you want to store PyBeam>
```

where you replace `<directory where you want to store PyBeam>` with the path to the directory where you want to have the `PyBeam` repository. This path could be `C:\Python` or `$HOME\Desktop`. Next, clone the repository with

```shell
git clone https://github.com/AAU-Python/PyBeam.git
```

You can now open VS Code in the PyBeam project with

```shell
code PyBeam
```

### Install the environment

Once inside VS Code, you can use the integrated terminal. To open an integrated terminal, press `F1`, type out "Terminal: Create new terminal", then press `Enter`.

To install the packages needed for PyBeam, run the following commands:

```shell
micromamba env create -f environment.yml -y
```

```shell
pip install -e .
```

Once done, activate your environment with

```shell
micromamba activate aau
```

and tell VS Code to use the `aau` environment by pressing `F1` then typing `Python: Select interpreter` and pressing `Enter`. You should be able to select the `aau` Python environment here.

### Open a notebook

To open a Jupyter notebook, go to the file explorer menu. It is usually located at the top left in the VS Code window, and the icon looks like two staced pieces of paper. Alternatively, you can open the file explorer menu by pressing `Ctrl` + `Shift` + `E` on Windows/Linux or `Command` + `Shift` + `E` on MacOS.

Find the `examples` folder, expand the dropdown, and then double click on any of the `.ipynb` files to open the notebook.

Once you have that open, select what environment to run the notebook in by pressing `F1`, typing "Notebook: Select Notebook kernel", pressing `Enter` and then selecting the `aau` environment.

You can now execute cells in the notebook with `Ctrl` + `Enter` on Windows/Linux or `Command` + `Enter` on MacOS. You can also execute + advance to the next cell with `Shift` + `Enter`.

## Getting updates from GitHub

If there are updates to PyBeam on GitHub, you can "pull" these updates with

```shell
git fetch
```

and then

```shell
git clone
```

Sometimes, changes to the environment are made. Simply run these commands again:

```shell
micromamba env create -f environment.yml -y
```

```shell
pip install -e .
```

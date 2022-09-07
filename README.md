# CICADA

A Collaborative, Informed and Context Aware Drawing Agent for Co-Creative Design

Before suffering through the installation process (that should pose no problem, but we all know how these things go) we recommend you try CICADA in a google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1rEnHMMBnK--qxXatwt9QEQexKsHZWizb?usp=sharing)

<br>

## Installation

### gcc version

<!-- Make sure your gcc version is 8 or lower -->
```
gcc --version
```

### Anaconda

Make sure you have Anaconda 3 installed and create an environment to install CICADA
```
conda update -n base -c defaults conda
conda env create -n cicada
conda activate cicada
```

### Install

Clone the repo, cd to it and run setup.sh (this can take a while)
```
git clone https://github.com/fibarrola/cicada.git
cd cicada
sh setup.sh
```

<br>

## Using CICADA

Once installed, it can be run as

```
$ python3 main.py
```

The following options can be used to change the default parameters

```
--svg_path <str>            path to svg partial sketch
--prompt <str>              what to draw
--neg_prompt <str>
--neg_prompt_2 <str>
--use_neg_prompts <bool>
--normalize_clip <bool>
--num_paths <int>           number of strokes to add
--canvas_h <int>            canvas height
--canvas_w <int>            canvas width
--max_width <int>           max px width
--num_iter <int>            maximum algorithm iterations
--w_points <float>          regularization parameter for Bezier point distance
--w_colors <float>          regularization parameter for color differences
--w_widths <float>          regularization parameter for width differences
--w_img <float>             regularization parameter for image L2 similarity
--w_geo <float>             regularization parameter for geometric similarity
--x0 <float>                coordinate for drawing area
--x1 <float>                coordinate for drawing area
--y0 <float>                coordinate for drawing area
--y1 <float>                coordinate for drawing area
--num_trials <int>          number of times to run the algorithm
--num_augs <int>            number of augmentations for computing semantic loss
--prune_ratio <float>       ratio of paths to be pruned out
--save_path <str>           subfolder for saving results
```


<!-- ![](repo_img/avocado_chair.gif?raw=true) -->

<br>

## Capability

CICADA is meant to give way to an interactive design process. Here we show an example of a human user drawing "An avocado chair", and using CICADA to aid her in making the drawing into a "An antique avocado chair". The interface is still under development and will be released shortly.

<p align="center">
<img src="https://github.com/fibarrola/cicada/blob/master/repo_img/avocado_chair.gif" width="448"/>
</p>

## co_creative_draw

To install, run
```
make install
```

For linting
```
make pretty
make lint
```

For testing
```
make test
```

For visual inspection of functionality
```
make inspection_test
```

Once installed, it can be run as

```
$ python3 main.py
```

The following options can be used to change the default parameters

```
--svg_path <str>            path to svg partial sketch
--prompt <str>         what to draw
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
--save_path <str>                 subfolder for saving results
```

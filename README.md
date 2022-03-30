## co_creative_draw

Once installed, it can be run as

```
$ python3 main.py
```

The following options can be used to change the default parameters

```
--svg_path SVG_PATH   path to svg partial sketch
--clip_prompt CLIP_PROMPT
                    what to draw
--neg_prompt NEG_PROMPT
--neg_prompt_2 NEG_PROMPT_2
--use_neg_prompts USE_NEG_PROMPTS
--normalize_clip NORMALIZE_CLIP
--num_paths NUM_PATHS
                    number of strokes to add
--canvas_h CANVAS_H   canvas height
--canvas_w CANVAS_W   canvas width
--max_width MAX_WIDTH
                    max px width
--num_iter NUM_ITER   maximum algorithm iterations
--w_points W_POINTS   regularization parameter for Bezier point distance
--w_colors W_COLORS   regularization parameter for color differences
--w_widths W_WIDTHS   regularization parameter for width differences
--w_img W_IMG         regularization parameter for image L2 similarity
--x0 X0               coordinate for drawing area
--x1 X1               coordinate for drawing area
--y0 Y0               coordinate for drawing area
--y1 Y1               coordinate for drawing area
--num_trials NUM_TRIALS
                    number of times to run the algorithm
```
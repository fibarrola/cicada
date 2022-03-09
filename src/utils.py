def save_data(time_str, params):
    with open('results/'+time_str+'.txt', 'w') as f:
        f.write('I0: ' +params.svg_path +'\n')
        f.write('prompt: ' +str(params.clip_prompt) +'\n')
        f.write('num paths: ' +str(params.num_paths) +'\n')
        f.write('num_iter: ' +str(params.num_iter) +'\n')
        f.write('w_points: '+str(params.w_points)+'\n')
        f.write('w_colors: '+str(params.w_colors)+'\n')
        f.write('w_widths: '+str(params.w_widths)+'\n')
        f.write('w_img: '+str(params.w_img)+'\n')
        f.close()
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

audio_dir = 'lrw_audio'
audio_path = os.path.join(dir_path, audio_dir)
audio_list = sorted(os.listdir(audio_path))

audio_train_path = os.path.join(audio_path, 'train')
audio_test_path = os.path.join(audio_path, 'test')
audio_val_path = os.path.join(audio_path, 'val')
if not os.path.exists(audio_train_path):
    os.mkdir(audio_train_path)
if not os.path.exists(audio_test_path):
    os.mkdir(audio_test_path)
if not os.path.exists(audio_val_path):
    os.mkdir(audio_val_path)

params_dir = 'lrw_shape_params'
params_path = os.path.join(dir_path, params_dir)
params_list = sorted(os.listdir(params_path))

params_train_path = os.path.join(params_path, 'train')
params_test_path = os.path.join(params_path, 'test')
params_val_path = os.path.join(params_path, 'val')
if not os.path.exists(params_train_path):
    os.mkdir(params_train_path)
if not os.path.exists(params_test_path):
    os.mkdir(params_test_path)
if not os.path.exists(params_val_path):
    os.mkdir(params_val_path)


for sample in audio_list:
    sample_path = os.path.join(audio_path, sample)

    label, tail = sample.split('_')
    idx, ext = tail.split('.')
    if int(idx) < 31:
        new_sample_path = os.path.join(audio_test_path, sample)
    elif int(idx) > 70:
        new_sample_path = os.path.join(audio_train_path, sample)
    else:
        new_sample_path = os.path.join(audio_val_path, sample)
    
    os.rename(sample_path, new_sample_path)

for sample in params_list:
    sample_path = os.path.join(params_path, sample)

    label, tail = sample.split('_')
    idx, ext = tail.split('.')
    if int(idx) < 31:
        new_sample_path = os.path.join(params_test_path, sample)
    elif int(idx) > 70:
        new_sample_path = os.path.join(params_train_path, sample)
    else:
        new_sample_path = os.path.join(params_val_path, sample)
    
    os.rename(sample_path, new_sample_path)
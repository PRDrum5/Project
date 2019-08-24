import os


def move_files(source, dest):
    f_list = sorted(os.listdir(source))

    for f in f_list:
        f_path = os.path.join(source, f)
        new_path = os.path.join(dest, f)
        os.rename(f_path, new_path)

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    audio_dir = 'lrw_audio'
    audio_path = os.path.join(dir_path, audio_dir)

    audio_train_path = os.path.join(audio_path, 'train')
    audio_test_path = os.path.join(audio_path, 'test')
    audio_val_path = os.path.join(audio_path, 'val')

    params_dir = 'lrw_shape_params'
    params_path = os.path.join(dir_path, params_dir)

    params_train_path = os.path.join(params_path, 'train')
    params_test_path = os.path.join(params_path, 'test')
    params_val_path = os.path.join(params_path, 'val')

    move_files(audio_train_path, audio_path)
    move_files(audio_test_path, audio_path)
    move_files(audio_val_path, audio_path)
    move_files(params_train_path, params_path)
    move_files(params_test_path, params_path)
    move_files(params_val_path, params_path)

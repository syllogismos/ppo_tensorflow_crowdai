import argparse
import tensorflow as tf
import os, shutil


def extract_snapshot(chk_dir, des_dir):
    """
    chkp_dir: name of the existing snapshot
    des_dir: name of the destination folder
    """
    scaler_file = chk_dir + '/scaler_latest'
    latest_pol_file = tf.train.latest_checkpoint(chk_dir, latest_filename='policy_checkpoint')
    latest_val_file = tf.train.latest_checkpoint(chk_dir, latest_filename='value_checkpoint')
    meta_file_pol = latest_pol_file + '.meta'
    meta_file_val = latest_val_file + '.meta'
    if not os.path.exists(meta_file_pol):
        meta_file_pol = chk_dir + '/policy-model-0.meta'
    if not os.path.exists(meta_file_val):
        meta_file_val = chk_dir + '/value-model-0.meta'

    des_path = os.path.abspath(des_dir)
    if os.path.exists(des_path):
        print('destination path exists')
        return
    os.makedirs(des_path)
    filenames = [
            scaler_file,
            meta_file_pol,
            latest_pol_file + '.data-00000-of-00001',
            latest_pol_file + '.index',
            meta_file_val,
            latest_val_file + '.data-00000-of-00001',
            latest_val_file + '.index',
            ]
    for filename in filenames:
        shutil.copy(filename, des_path)

    basename_pol = os.path.basename(latest_pol_file)
    basename_val = os.path.basename(latest_val_file)
    build_checkpoint_file(des_dir, basename_pol, 'policy_checkpoint')
    build_checkpoint_file(des_dir, basename_val, 'value_checkpoint')
    return

def build_checkpoint_file(des_dir, basename, checkpoint):
    f = open(des_dir + '/' + checkpoint, 'w')

    f.write('model_checkpoint_path: "%s"'%basename)
    f.write('\n')
    f.write('all_model_checkpoint_paths: "%s"'%basename)
    f.close()
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=str)
    parser.add_argument("-f", "--folder_name", type=str, help="name of the destination folder")
    args = parser.parse_args()
    extract_snapshot(args.snapshot, args.folder_name)


if __name__ == '__main__':
    main()

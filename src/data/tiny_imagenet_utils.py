import numpy as np
import pandas as pd


# Source: https://github.com/rmccorm4/Tiny-Imagenet-200/blob/master/networks/data_utils.py#L38
def load_tiny_imagenet(path, wnids_path, resize='False', num_classes=200, dtype=np.float32):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.

    Returns: A tuple of
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    """
    # First load wnids
    # wnids_file = os.path.join(wnids_path, 'wnids' + str(num_classes) + '.txt')
    wnids_file = 'wnids' + '.txt'

    with open(os.path.join(path, wnids_file), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    # words_file = os.path.join(wnids_path, 'words' + str(num_classes) + '.txt')
    words_file = os.path.join(wnids_path, 'words' + '.txt')

    with open(words_file, 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    img_path_ls_train = []

    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        if resize.lower() == 'true':
            X_train_block = np.zeros((num_images, 3, 32, 32), dtype=dtype)
        else:
            X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)

        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            # img = imageio.v2.imread(img_file)
            img_path_ls_train.append(img_file)
        #
        #     if resize.lower() == 'true':
        #         img = scipy.misc.imresize(img, (32, 32, 3))
        #     if img.ndim == 2:
        #         ## grayscale file
        #         if resize.lower() == 'true':
        #             img.shape = (32, 32, 1)
        #         else:
        #             img.shape = (64, 64, 1)
        #     X_train_block[j] = img.transpose(2, 0, 1)

        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_path_ls_val = []

        img_files = []
        val_wnids = []
        for line in f:
            # Select only validation images in chosen wnids set
            if line.split()[1] in wnids:
                img_file, wnid = line.split('\t')[:2]
                img_files.append(img_file)
                val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

        if resize.lower() == 'true':
            X_val = np.zeros((num_val, 3, 32, 32), dtype=dtype)
        else:
            X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)

        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img_path_ls_val.append(img_file)

            # img = imageio.v2.imread(img_file)
            # if resize.lower() == 'true':
            #     img = scipy.misc.imresize(img, (32, 32, 3))
            # if img.ndim == 2:
            #     if resize.lower() == 'true':
            #         img.shape = (32, 32, 1)
            #     else:
            #         img.shape = (64, 64, 1)
            #
            # X_val[i] = img.transpose(2, 0, 1)
    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    # img_files = os.listdir(os.path.join(path, 'test', 'images'))
    # X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    # for i, img_file in enumerate(img_files):
    #     img_file = os.path.join(path, 'test', 'images', img_file)
    #     img = imageio.v2.imread(img_file)
    #     if img.ndim == 2:
    #         img.shape = (64, 64, 1)
    #     X_test[i] = img.transpose(2, 0, 1)

    # y_test = None
    # y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    # if os.path.isfile(y_test_file):
    #     with open(y_test_file, 'r') as f:
    #         img_file_to_wnid = {}
    #         for line in f:
    #             line = line.split('\t')
    #             img_file_to_wnid[line[0]] = line[1]
    #     y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    #     y_test = np.array(y_test)
    #
    # return class_names, X_train, y_train, X_val, y_val, X_test, y_test
    return class_names, X_train, y_train, X_val, y_val, img_path_ls_train, img_path_ls_val


import requests
import zipfile
import os


def download_and_extract(url, target_folder):
    # Create target folder if it doesn't exist
    data_dir = os.path.join(target_folder, "tiny-tiny_imagenet-200")
    if not os.path.exists(data_dir):
        os.makedirs(target_folder, exist_ok=True)

        # Download the zip file
        response = requests.get(url, stream=True)
        zip_filename = os.path.join(target_folder, os.path.basename(url))

        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(target_folder)

        # Remove the zip file after extraction
        os.remove(zip_filename)


def download_and_extract_tiny_imagenet(target_folder):
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Download and extract the dataset
    download_and_extract(dataset_url, target_folder)


def generate_tiny_imagenet_data(data_dir):
    class_names, X_train, y_train, X_val, y_val, img_path_ls_train, img_path_ls_val = load_tiny_imagenet(data_dir,
                                                                                                         data_dir)

    train_df_ = pd.DataFrame(
        [{"img_path": img_path, "y_label": y_} for img_path, y_ in zip(img_path_ls_train, y_train)])
    test_df = pd.DataFrame([{"img_path": img_path, "y_label": y_} for img_path, y_ in zip(img_path_ls_val, y_val)])

    test_df = test_df.sort_values(["y_label", "img_path"])

    data_idx = np.arange(len(train_df_))
    np.random.seed(100)
    np.random.shuffle(data_idx)

    train_count = int(0.9 * len(train_df_))
    train_idx = data_idx[:train_count]
    valid_idx = data_idx[train_count:]

    train_df = train_df_.loc[train_idx, :]
    valid_df = train_df_.loc[valid_idx, :]

    train_df.to_csv(os.path.join(data_dir, "train_tiny_imagenet200.csv"), index=None)
    valid_df.to_csv(os.path.join(data_dir, "valid_tiny_imagenet200.csv"), index=None)
    test_df.to_csv(os.path.join(data_dir, "test_tiny_imagenet200.csv"), index=None)

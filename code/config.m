function conf = config()

conf = struct();

%% Directories

conf.root_path = '/home/kazeka/data/fc_tuning/';
conf.cache_path = '/home/kazeka/data/fc_tuning/cache';
conf.image_path = '/home/rnd/src/bloom/data-37/256/';

%% Training parameters
conf.preprocessing.crop_padding = 15;
conf.preprocessing.square_size = 256;
conf.preprocessing.num_to_load = 255;
conf.preprocessed_image_file = [conf.cache_path, '/training_images.mat'];

path_triplet = '/home/kazeka/data/fc_tuning/cache';

% for multilabel pairs batchsize = 128
conf.training_set_path_multilabel_m128 = [path_triplet, '/training_set_multilabel_m128.lmdb'];

% for multilabel pairs batchsize = 128*2 = 256
conf.training_set_path_multilabel_m256 = [path_triplet, '/training_set_multilabel_m256.lmdb'];

% for multilabel pairs batchsize = 128*3 = 384
conf.training_set_path_multilabel_m384 = [path_triplet, '/training_set_multilabel_m384.lmdb'];

% for debuggin,
conf.training_imageset_path = [path_triplet, '/training_imageset.lmdb'];

conf.training_set_path_triplet = [path_triplet, '/training_set_triplet.lmdb'];
conf.validation_set_path_triplet = [path_triplet, '/validation_set_triplet.lmdb'];
conf.validation_set_path = [path_triplet, '/validation_set.lmdb'];

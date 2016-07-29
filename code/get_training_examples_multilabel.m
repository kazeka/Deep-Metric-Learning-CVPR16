function [image_id_pairs, labels] = get_training_examples_multilabel(...
                                                    mode, batchsize)

% fix the seed for randperm
randn('state', 1);
rand('state', 1);

conf = config;

dict = load([conf.root_path, 'dict.mat']).d;
train_images = load([conf.root_path, 'train_images.mat']).train_images;
val_images = load([conf.root_path, 'val_images.mat']).val_images;

class_id_breakpoint = 31;
num_classes = length(dict);

num_images = length(train_images);
image_list = train_images;
class_list = 1:class_id_breakpoint;

num_training_images = length(train_images);
% assert(num_training_images == max(dict{class_id_breakpoint}));

% Build reverse look up of dictionary to get class ids of randomly sampled
% negative images
dict_reverse = struct();
keys = 1:length(dict);
for i = 1:length(keys)
    this_class_idx = keys(i);
    this_class_img_ids = dict{this_class_idx};
    for j = 1:length(this_class_img_ids)
        this_img_idx = this_class_img_ids(j);
        dict_reverse.(num2str(this_img_idx)) = this_class_idx;
    end
end

%% positive examples

% count how many positive examples can be generated exhaustively.
% num_pos_pairs = 0;
% for class_id = class_list
%     this_class_num_images = length(dict{class_id});
%     fprintf('class: %d, num images: %d\n', class_id, this_class_num_images);
%     if this_class_num_images > 1
%         num_pos_pairs = num_pos_pairs + nchoosek(this_class_num_images, 2);
%     else
%         error('[ERROR] class %d has only 1 image\n', class_id);
%     end
%     num_pos_pairs = num_pos_pairs + this_class_num_images;
% end
% fprintf('mode %s has %d pos pairs\n', mode, num_pos_pairs);

% downsample size: examples per image
epi = 25;

% construct the positive set.
pos_pairs = zeros(epi * num_images, 2);
pos_class = zeros(epi * num_images, 2);
insert_idx = 1;
for class_id = class_list
    image_ids = dict{class_id};
    this_class_num_images = length(image_ids);
    num_combinations = epi * this_class_num_images;  % nchoosek(this_class_num_images, 2);
    total_num_combinations = nchoosek(this_class_num_images, 2);
    sample_ids = randperm(total_num_combinations)(1:num_combinations);
    pos_pairs(insert_idx:insert_idx + num_combinations-1, :) = ...
        nchoosek(image_ids, 2)(sample_ids, :);
    pos_class(insert_idx:insert_idx + num_combinations-1, :) = class_id;
    insert_idx = insert_idx + num_combinations;
    % add self pairs
    pos_pairs(insert_idx : insert_idx + this_class_num_images-1, :) = ...
        repmat(image_ids', 1, 2);
    pos_class(insert_idx : insert_idx + this_class_num_images-1, :) = class_id;
    insert_idx = insert_idx + this_class_num_images;
end
% assert(num_pos_pairs == size(pos_pairs, 1), 'dim mismatch.');

%% negative examples
num_negs_per_image = epi + 2; % ceil(num_pos_pairs / num_images);
neg_pairs = zeros(num_negs_per_image * num_images, 2);
neg_class = zeros(num_negs_per_image * num_images, 2);
insert_idx = 1;
for class_id = class_list
    fprintf('class: %d\n', class_id);
    image_ids = dict{class_id};
    sampled_neg_ids = negative_sampler(num_images, image_ids, ...
        num_negs_per_image);
    fprintf('sizes: %d, %d\n', size(neg_pairs), size(sampled_neg_ids));
    neg_pairs(insert_idx:insert_idx + length(sampled_neg_ids)-1, :) = ...
        sampled_neg_ids;
    neg_class(insert_idx:insert_idx + length(sampled_neg_ids)-1, 1) = ...
        class_id;
    for j = 1:length(sampled_neg_ids)
        neg_class(insert_idx+j-1,2) = dict_reverse.(num2str(sampled_neg_ids(j,2)));
        assert(neg_class(insert_idx+j-1)) == dict_reverse.(num2str(sampled_neg_ids(j,1)));
    end

    insert_idx = insert_idx + length(sampled_neg_ids);
end

%fprintf('neg_pairs: %d, %d\n', length(neg_pairs), num_negs_per_image * num_images);
% assert(length(neg_pairs) == ...
%     num_negs_per_image * num_images, 'dim mismatch.');
assert(length(neg_pairs) > length(pos_pairs));

%% Assemble and shuffle

% make the number of data divisible by the batchsize 256
%   do this by deleting some data from neg pairs b/c more negs than pos

assert(mod(batchsize, 2)==0);

num_total = length(pos_pairs) + length(neg_pairs);
num_to_delete = num_total - floor(num_total/(batchsize/2))*(batchsize/2);
fprintf('Deleted %d data pairs from negatives for batch divisibility\n', ...
        num_to_delete);
groundset = 1:length(neg_pairs);
inds_to_delete = randsample(groundset, num_to_delete, false);
neg_pairs(sort(inds_to_delete, 'ascend'), :) = [];
neg_class(sort(inds_to_delete, 'ascend'), :) = [];

image_id_pairs = [pos_pairs; neg_pairs];
assert(mod(length(image_id_pairs), batchsize/2) == 0);

labels = [pos_class; neg_class];
assert(length(labels) == length(image_id_pairs));

perm = randperm(size(image_id_pairs,1));
image_id_pairs = image_id_pairs(perm,:);
labels = labels(perm,:);



function neg_pair_ids = negative_sampler(...
    num_images, image_ids, num_negs_per_image)

% excluding image_ids,
%   sample (num_negs_per_image * num_views) number of negatives

num_neg_samples_required = num_negs_per_image * length(image_ids);

groundset = 1:num_images;
groundset(image_ids) = [];

fprintf('numel: %d, %d\n', numel(groundset), num_neg_samples_required);
% sample_ids = randsample(groundset, num_neg_samples_required, false); % nchoosek(groundset, num_neg_samples_required); % (randperm(size(numel(groundset) * num_neg_samples_required, 1)),:);
sample_ids = zeros(1, num_neg_samples_required);
for i = 1:num_neg_samples_required
  sample_ids(1, i) = groundset(unidrnd(length(groundset)));
end

left_image_ids = repmat(image_ids, num_negs_per_image, 1);

neg_pair_ids = [reshape(left_image_ids, [], 1), sample_ids'];

function gen_splits

conf = config;

% conf.root_path = '/scail/scratch/group/cvgl/hsong/Deep-Lifting-for-Metric-Learning-CVPR/code/ebay/';
% conf.cache_path = '/scail/scratch/group/cvgl/hsong/Deep-Lifting-for-Metric-Learning-CVPR/code/ebay/cache';
% conf.image_path = '/cvgl/group/Ebay_Dataset/';

%% generate splits
[image_ids, class_ids, superclass_ids, path_list] = ...
    textread('/data/stanford_products/Stanford_Online_Products/Ebay_train.txt', '%d %d %d %s',...
    'headerlines', 1);

train_images = {};
dict = struct();

for i = 1:length(image_ids)
    imageid = image_ids(i);
    classid = class_ids(i);
    filename = path_list{i};

    fprintf('%d/%d, classid= %d, filename= %s\n', ...
        i, length(image_ids), classid, filename);

    train_images{end+1} = filename;

    % hash it
    if isfield(dict, num2str(classid))
        dict.(num2str(classid)) = [dict.(num2str(classid)), imageid];
    else
        dict.(num2str(classid)) = [imageid];
    end
end

[image_ids, class_ids, superclass_ids, path_list] = ...
    textread('/data/stanford_products/Stanford_Online_Products/Ebay_test.txt', '%d %d %d %s',...
    'headerlines', 1);

val_images = {};
for i = 1:length(image_ids)
    imageid = image_ids(i);
    classid = class_ids(i);
    filename = path_list{i};

    fprintf('%d/%d, classid= %d, filename= %s\n', ...
        i, length(image_ids), classid, filename);

    val_images{end+1} = filename;

    % hash it
    if isfield(dict, num2str(classid))
        dict.(num2str(classid)) = [dict.(num2str(classid)), imageid];
    else
        dict.(num2str(classid)) = [imageid];
    end
end

d = {};
for i = 1:length(fieldnames(dict))
    d{i} = dict.(num2str(i));
end

save('-text', [conf.root_path, 'dict.mat'], 'd');
save('-text', [conf.root_path, 'train_images.mat'], 'train_images');
save('-text', [conf.root_path, 'val_images.mat'], 'val_images');

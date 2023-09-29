% 1. Read all colar images and converted to gray-scale images
imds = imageDatastore("CroppedYale/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".pgm");
fileNum = size(imds.Files, 1);

% 1.1 convert rgb photos to gray-scale
for i = 1:fileNum
    [img,info] = readimage(imds, i);
    color = imfinfo(info.Filename);
    if (color.ColorType == "truecolor")
        img = rgb2gray(img);
    end
end

% 1.2 delete photos not in the size of 192*168 from imds
photoSize = 192*168;
for i = 1:fileNum
    tmp = readimage(imds,i);
    tmp = reshape(tmp, 1, []);
    if (length(tmp) ~= photoSize)
        [tmp ,info]= readimage(imds,i);
        delete (info.Filename);
    end
end
imds = imageDatastore("CroppedYale/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".pgm");
fileNum = size(imds.Files, 1);

% 2. Randomly split images into trainging set / testing set
[trainSet, testSet] = splitEachLabel(imds, 35, "randomized");
trainNum = numel(trainSet.Files);
testNum = numel(testSet.Files);

% 2.1 store files in imds trainSet into trainImg
photoSize = 192*168;
trainImg = zeros(trainNum, photoSize);
for i=1:trainNum
     tmp = readimage(trainSet,i);
     tmp = reshape(tmp, 1, []);
     trainImg(i,:) = tmp;
end

% 2.2 store files in imds testSet into testImg
testImg = zeros(testNum, photoSize);
for i=1:testNum
     tmp = readimage(testSet,i);
     tmp = reshape(tmp, 1, []);
     testImg(i,:) = tmp;
end

% 3. Find NN for each test image
resIdx = knnsearch(trainImg, testImg,"Distance","cityblock");
% 3.1 show 5 NN result samples
for i=1:5
    disp("test file:");
    disp(testSet.Files(1));
    disp("train file:");
    disp(trainSet.Files(resIdx(i)));
end

% 4. Caculate the accuracy for NN method
correctCnt = 0;
for i=1:testNum
    if (trainSet.Labels(resIdx(i)) == testSet.Labels(i))
        correctCnt = correctCnt + 1;
    end
end
acc = correctCnt / testNum;
disp("SAD acc:" + acc);

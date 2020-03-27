import cv2
import numpy as np
import os
sift = cv2.xfeatures2d.SIFT_create()

class FeatureMatchingFramework:

    def __init__(self, directory, object_names, dictionary_size=64):
        """
        Prepares framework for use, by calculating feature descriptors of all images in a given directory
        :param directory: string specifying relative path of directory containing training images
        :param object_names: dictionary such that object_names[file_name] = name of object
        :param dictionary_size: int number of KMeans Clusters for Bag of Words
        """

        self.object_names = object_names

        self.bow_trainer = cv2.BOWKMeansTrainer(dictionary_size)

        self.training_kps = {}  # stores keypoints of training images
        self.training_descs = {}  # stores descriptors of training images
        self.training_images = {}  # stores training images

        max_width = 0  # keep track of the maximum width of a training image

        image_files = os.listdir(directory)  # get all training images

        # load each image, save to self.training_images, and find the largest image width
        for image_file in image_files:
            image = cv2.imread(directory + image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            max_width = max(image.shape[1], max_width)
            self.training_images[image_file] = image

        # resize each image so that they are the same width, calculate sift keypoints and descriptors
        for image_file in image_files:
        
            image = self.training_images[image_file]

            # resize image
            height = image.shape[0]  # keep original height
            width = max_width
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            self.training_images[image_file] = image

            # get sift keypoints and descriptors
            kp = sift.detect(image, None)
            self.training_kps[image_file] = kp
            kp, des = sift.compute(image, kp)
            self.training_descs[image_file] = des

            # add to the BoW trainer
            self.bow_trainer.add(des) 

        # calculate clusters and put their centers in the dictionary
        dictionary = self.bow_trainer.cluster()

        # create BOWImgDescriptorExtractor using the dictionary
        self.bow_extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
        self.bow_extractor.setVocabulary(dictionary)

        """
        compute the BoW representation of each train image and
        save in X_train[i], then save the corresponding file name in y_train[i]
        """
        self.X_train = []
        self.y_train = []
        for image_file in image_files:
            image = self.training_images[image_file]
            bow_feats = self.bow_extractor.compute(image, sift.detect(image))[0].tolist()
            self.X_train.append(bow_feats)
            self.y_train.append(image_file)

    @staticmethod
    def mse(arr_1, arr_2):
        """
        Return the MSE between two arrays
        :param arr_1: array 1
        :param arr_2: array 2
        :return: MSE between arr_1 and arr_2
        """
        return np.mean(np.square(np.asarray(arr_1)-np.asarray(arr_2)))
    
    @staticmethod
    def get_matches(des1, des2):
        """
        Get the feature matches between two images given a list of their SIFT descriptors
        :param des1: SIFT descriptors
        :param des2: SIFT descriptors
        :return: nearest neighbour matches between des1 and des2
        """
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)  # make FLANN searcher
        return flann.knnMatch(des1, des2, k=2)  # return matches

    @staticmethod
    def get_location_of_object_in_image(image, matches, kp):
        """
        Returns estimated leftmost, rightmost, top-most and bottom-most coordinates of object in image
        :param image: image in which to get location of object
        :param matches: list of matches between the image and a training image of the object
        :param kp: the SIFT keypoints of the image
        :return:
        """
        min_train_x = image.shape[1]
        min_train_y = image.shape[0]
        max_train_x = 0
        max_train_y = 0

        count = 0  # number of matches that pass Lowe's ratio test

        for i, (m, n) in enumerate(matches):
            # only consider the match if it passes Lowe's ratio test
            if m.distance < 0.7 * n.distance:
                count += 1
                train_coord = cv2.KeyPoint_convert([kp[m.trainIdx]])[0]
                min_train_x = min(train_coord[0], min_train_x)
                max_train_x = max(train_coord[0], max_train_x)
                min_train_y = min(train_coord[1], min_train_y)
                max_train_y = max(train_coord[1], max_train_y)

        """
        if no matches pass Lowe's ratio test, we set the leftmost, rightmost, top-most and bottom-most coords as the
        image extremities
        """
        if count <= 1:
            min_train_x = 0
            min_train_y = 0
            max_train_x = np.shape(image)[1]
            max_train_y = np.shape(image)[0]

        # we subtract 1, one from each value so that coordinates start at 0 not 1
        return int(min_train_x - 1), int(min_train_y - 1), int(max_train_x - 1), int(max_train_y - 1)
 
    def get_nearest_neighbour_prediction(self, image, kp):
        """
        Return file name of image whose BoW is most similar to the BoW of 'image'
        :param image: A test image
        :param kp: Keypoints of the image
        :return: File name of image whose BoW is most similar to the BoW of 'image'
        """
        bow_feats = self.bow_extractor.compute(image, kp)[0].tolist()
        distances = [self.mse(x, bow_feats) for x in self.X_train]
        return self.y_train[distances.index(min(distances))]

    def predict(self, images):
        """
        Return an array such that each element is the name of the object whose training image has the nearest neighbour
        BoW representation of each image in the 'images' array
        :param images: array/list of images
        :return: array such that each element is the name of the object whose training image has the nearest neighbour
        BoW representation of the corresponding image in the 'images' array
        """
        predictions = []
        for image in images:
            kp = sift.detect(image)
            if len(kp) < 1:
                # if no keypoints are detected we predict that there is no object
                predictions.append(None) 
            else:
                predictions.append(self.object_names[self.get_nearest_neighbour_prediction(image, kp)])
        return np.asarray(predictions)

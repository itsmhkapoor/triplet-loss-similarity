# Triplet Loss for Similarity
In this scenario, three images are given 'anchor, positive and negative' where the positive image is similar to the anchor image and the negative image is dissimilar to the anchor. In the test case, given a triplet <A, B, C> if A is similar to B than C then we predict 1 else 0. 

## Requirements
Execute the following command to install dependencies.
```bash
pip install -r requirements.txt
```

## Dataset
The model was trained and tested on a private dataset. The training triplets were given in a .txt file in the format 'anchor_image_name positive_image_name negative_image_name' and so were the test triplets in another file. As a pre-processing step all images are resized to 224x224 (input size of ResNet50).

## Model
ResNet 50 was chosen as the base network for feature extraction from given images. It is a pre-trained network that classifies images into 1000 categories. Our task is not a classification task, hence we do not use the full ResNet-50 architecture. ResNet-50 till the second last layer (‘avg_pool’ layer) is used for feature extraction (base model), which is pre-trained on ImageNet dataset. Also these weights are not modified during training. To the base model, an encoding network is attached that converts the extracted features to 10 dimensional encodings. This network consists of a dense layer having 512 nodes followed by output of encoding size (10). Dropouts are added to avoid overfitting and the dense layers have ‘he uniform’ weight initializers. A normalizing layer was defined at the output, which performs L2 normalization of the encodings. So the overall model has 3 inputs but the weights of the model are shared among the inputs. This sharing is done so that same inputs produce the same encodings. The final output is a vector which is the concatenation of all 3 image encodings. 

## Triplet Loss
Triplet loss minimizes the distance between the anchor and positive encoding while maximizing the distance between anchor and negative encoding. The distance metric here can be L1, L2 or cosine similarity (from keras). Also for the predictions, a similar approach was used, where if the distance between anchor and positive encoding is less than between anchor and negative encoding, then 1 is predicted and vice versa.
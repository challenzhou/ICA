// Makeup.cpp : Defines the entry point for the console application.
//
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define IMG_COUNT 10
//#define IMG_PATH "./element/Baby"
#define IMG_PATH "/data/dataset/face/ATT/s2"

#define COMPONENT_SIZE 4

#define ITER_COUNT 100

std::string numberToString(int number, unsigned int count_bytes)
{
  std::string out = std::to_string(number);

  while (out.length() < count_bytes) {
    out = "0" + out;
  }

  return out;
}

bool getImageSet(std::string path, int count, cv::Mat &imageSet, cv::Size &size)
{
  if (path.empty() || count <= 0)
    return false;

  for (int i=0; i<count; i++)
  {
    std::string img_path = path + "/" + numberToString(i+1, 1) + ".pgm";
	  cv::Mat image = cv::imread(img_path);
    if (image.empty())
    {
      std::cout << "Image path:"<< img_path <<" is not correct!!!" << std::endl;
      return 0;
    } else {
      std::cout << img_path <<" size:"<< image.size() << std::endl;
	    cv::cvtColor(image, image, CV_BGR2GRAY);
      image.convertTo(image, CV_32F);

      if (!size.width || !size.height)
      {
        size = image.size();
      }

      image = image.reshape(1, image.rows*image.cols);
      std::cout << img_path <<" reshaped size:"<< image.size() <<" image rows:"<< image.rows << std::endl;
      if (imageSet.empty())
        imageSet = image;
      else
        cv::hconcat(imageSet, image, imageSet);
    }
  }

  return true;
}

bool whitenImageSet(cv::Mat &imageset, cv::Mat &imagemap, cv::Mat &imagemean,cv::Size &size)
{
  if (imageset.empty() || size.area() <= 0)
    return false;

  cv::PCA pca(imageset, cv::Mat(), cv::PCA::DATA_AS_COL, 1.0f);
  std::cout << "dataset pca eigvectors Size: " << pca.eigenvectors.size() << std::endl;
  std::cout << "dataset pca eigenvalues Size: " << pca.eigenvalues.size() << std::endl;
  std::cout << "dataset pca mean Size: " << pca.mean.size() << std::endl;

  // memset mean to make imagemap as zero average samples 
  imagemean = pca.mean;
  pca.mean = cv::Mat::zeros(pca.mean.size(), pca.mean.type());

  cv::Mat eigenscale, mapvector;
  cv::sqrt(pca.eigenvalues, eigenscale);
  eigenscale = 1.0f/eigenscale;

  mapvector = pca.project(imageset);
  cv::Mat scaletmp = cv::repeat(eigenscale, mapvector.rows/eigenscale.rows, mapvector.cols/eigenscale.cols);
  mapvector = mapvector.mul(scaletmp);
  imagemap = pca.backProject(mapvector);

  return true;
}

bool nmf(cv::Mat &imageset, cv::Mat &basis, cv::Mat &trans, uint32_t iters)
{
  if (imageset.empty() || iters <= 0)
    return false;

  basis = cv::Mat(imageset.rows, COMPONENT_SIZE, imageset.type());
  trans = cv::Mat(COMPONENT_SIZE, imageset.cols, imageset.type());
  cv::randu(basis, cv::Scalar(0.0f), cv::Scalar(255.0f));
  cv::randu(trans, cv::Scalar(0.0f), cv::Scalar(1.0f));
  std::cout << "basis Size:"<< basis.size() << std::endl;
  std::cout << "trans Size:"<< trans.size() << std::endl;

  while (iters-- > 0)
  {
    cv::Mat basis_top = imageset*trans.t();
    cv::Mat basis_bottom = basis*trans*trans.t();
    basis_bottom = 1.0f/basis_bottom;
    basis = basis.mul(basis_top);
    basis = basis.mul(basis_bottom);
 
    cv::Mat trans_top = basis.t()*imageset; 
    cv::Mat trans_bottom = basis.t()*basis*trans; 
    trans_bottom = 1.0f/trans_bottom;
    trans = trans.mul(trans_top);
    trans = trans.mul(trans_bottom);

    cv::Mat thresh = trans > 0.2f;
    thresh.convertTo(thresh, CV_8U);
    cv::normalize(thresh, thresh, 0.0f, 1.0f, cv::NORM_MINMAX);
    thresh.convertTo(thresh, CV_32F);
    trans = trans.mul(thresh);

    cv::Mat trans_sum;
    reduce(trans, trans_sum, 0, cv::REDUCE_SUM, CV_32F);
    trans_sum = 1.0f/trans_sum;

    cv::Mat trans_regular;
    for (int i=0; i<trans.cols; i++)
    {
   //   trans.col(i) *= trans_sum.at<float>(i);
    }

  }

  return true;
}

void show_subimage(std::string caption, cv::Mat &mat, cv::Size &size)
{

  cv::Mat base_show;
  for (int i=0; i<mat.cols; i++)
  {
    cv::Mat base_item= mat.col(i).clone();
//    base_item.convertTo(base_item, CV_8U);
    base_item = base_item.reshape(1, size.height);
//    base_show.push_back(base_item);
    cv::normalize(base_item, base_item, 0.0f, 1.0f, cv::NORM_MINMAX);
    if (base_show.empty())
      base_show = base_item;
    else
      cv::hconcat(base_show, base_item, base_show);
  }
  imshow(caption, base_show);
}

int main(int argc, char* argv[])
{
	// Get image set from directory
  cv::Mat imageSet;
  cv::Mat imageMap, imageMean;
  cv::Size imageSize;
  
  if (!getImageSet(IMG_PATH, IMG_COUNT, imageSet, imageSize))
    return -1;

  if (!whitenImageSet(imageSet, imageMap, imageMean, imageSize))
    return -1;

  cv::Mat Basis;
  cv::Mat Trans;
  if (!nmf(imageSet, Basis, Trans, ITER_COUNT))
   return -1;

#if 0
  cv::Mat Basis(imageSet.rows, COMPONENT_SIZE, imageSet.type());
  cv::Mat Trans(COMPONENT_SIZE, imageSet.cols, imageSet.type());
  cv::randu(Basis, cv::Scalar(0.0f), cv::Scalar(255.0f));
  cv::randu(Trans, cv::Scalar(0.0f), cv::Scalar(1.0f));
  std::cout << "Basis size:"<< Basis.size() << std::endl;
  std::cout << "Trans size:"<< Trans.size() << std::endl;

  int loop = ITER_COUNT;
  while (loop-- > 0)
  {
    cv::Mat basis_top = imageSet*Trans.t();
    cv::Mat basis_bottom = Basis*Trans*Trans.t();
    basis_bottom = 1.0f/basis_bottom;
    Basis = Basis.mul(basis_top);
    Basis = Basis.mul(basis_bottom);
 
    cv::Mat trans_top = Basis.t()*imageSet; 
    cv::Mat trans_bottom = Basis.t()*Basis*Trans; 
    trans_bottom = 1.0f/trans_bottom;
    Trans = Trans.mul(trans_top);
    Trans = Trans.mul(trans_bottom);

    cv::Mat thresh = Trans > 0.2f;
    thresh.convertTo(thresh, CV_8U);
    cv::normalize(thresh, thresh, 0.0f, 1.0f, cv::NORM_MINMAX);
    thresh.convertTo(thresh, CV_32F);
    Trans = Trans.mul(thresh);

    cv::Mat trans_sum;
    reduce(Trans, trans_sum, 0, cv::REDUCE_SUM, CV_32F);
    trans_sum = 1.0f/trans_sum;

    cv::Mat trans_regular;
    for (int i=0; i<Trans.cols; i++)
    {
   //   Trans.col(i) *= trans_sum.at<float>(i);
    }

  }
#endif

  show_subimage("Basis", Basis, imageSize);

  cv::Mat fit = Basis * Trans;
  show_subimage("Fit", fit, imageSize);

  show_subimage("Origin", imageSet, imageSize);
  show_subimage("Whiten", imageMap, imageSize);

  std::cout << "Trans:" << std::endl;
  std::cout << Trans << std::endl;
  std::cout << "Trans Sum:" << std::endl;
  cv::Mat Trans_sum;
  reduce(Trans, Trans_sum, 1, cv::REDUCE_SUM, CV_32F);
  std::cout << Trans_sum << std::endl;

  while(true)
  {
    int key = cv::waitKey(10);
    if (key == ' ')
      break;
  }

	return 0;
}


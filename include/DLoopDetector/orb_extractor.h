/**
 * File: orb_extractor.h
 * Date: December 2018
 * Author: Bill
 * Description: orb extractor
 * License: see the LICENSE.txt file
 */

#ifndef __ORB_EXTRACTOR__
#define __ORB_EXTRACTOR__

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <DLoopDetector/feature_extractor.h>

using namespace DBoW2;
using namespace DVision;
using namespace std;

namespace DLoopDetector {

/// This functor extracts BRIEF descriptors in the required format
class ORBExtractor: public FeatureExtractor<FORB::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<FORB::TDescriptor> &descriptors) const;

  /**
   * Creates the orb extractor
   */
  ORBExtractor();

private:
  void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) const;

  /// ORB descriptor extractor
  cv::Ptr<cv::ORB> orb_ptr;
};


ORBExtractor::ORBExtractor()
{
  orb_ptr = cv::ORB::create();  
}


void ORBExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<FORB::TDescriptor> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 10; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);

  std::sort(keys.begin(), keys.end(), [](const cv::KeyPoint& pt1, const cv::KeyPoint& pt2){
        return pt1.response > pt2.response;
  });
  keys.erase(
        keys.begin()+max_feat, keys.end());  

  cv::Mat descriptorsInMat;
  orb_ptr->compute(im, keys, descriptorsInMat);
  changeStructure(descriptorsInMat, descriptors);  
}

void ORBExtractor::changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) const
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

} // namespace DLoopDetector

#endif

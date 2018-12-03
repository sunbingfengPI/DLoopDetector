/**
 * File: brief_extractor.h
 * Date: December 2018
 * Author: Bill
 * Description: brief extractor
 * License: see the LICENSE.txt file
 */

#ifndef __BRIEF_EXTRACTOR__
#define __BRIEF_EXTRACTOR__

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
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<FBrief::TDescriptor> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};


BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}


void BriefExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<FBrief::TDescriptor> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 10; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);

  std::sort(keys.begin(), keys.end(), [](const cv::KeyPoint& pt1, const cv::KeyPoint& pt2){
        return pt1.response > pt2.response;
  });
  keys.erase(
        keys.begin()+max_feat, keys.end());  

  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}

} // namespace DLoopDetector

#endif


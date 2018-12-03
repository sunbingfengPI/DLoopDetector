/**
 * File: custom_loop_detector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: custom implementation of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __CUSTOM_LOOP_DETECTOR__
#define __CUSTOM_LOOP_DETECTOR__

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include <DLoopDetector/DLoopDetector.h>
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

#include <opencv2/imgproc/imgproc_c.h>

#include <DLoopDetector/feature_extractor.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

static const int IMAGE_W = 1280; // image size
static const int IMAGE_H = 720;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: BriefVocabulary)
/// @param TDetector detector class (e.g: BriefLoopDetector)
/// @param TDescriptor descriptor class (e.g: bitset for Brief)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class CustomLoopDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  CustomLoopDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height);
    
  ~CustomLoopDetector(){}

  /**
   * Detect if loop exist
   * @param name demo name
   * @param extractor functor to extract features
   */
  DetectionResult detect(const std::vector<cv::KeyPoint> &keys, const std::vector<TDescriptor> &descriptors);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs, 
    std::vector<double> &ys) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_posefile;
  int m_width;
  int m_height;

  TVocabulary voc;
  std::shared_ptr<TDetector> pdetector;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
CustomLoopDetector<TVocabulary, TDetector, TDescriptor>::CustomLoopDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height),
    voc(m_vocfile)
{
  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = 0.3; // nss threshold
  params.k = 3; // a loop must be consistent with 1 previous matches
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 2; // use two direct index levels  
  
  // Initiate loop detector with the vocabulary 
  cout << "Processing sequence..." << endl;
  pdetector = std::make_shared<TDetector>(voc, params);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
DetectionResult CustomLoopDetector<TVocabulary, TDetector, TDescriptor>::detect
                      (const std::vector<cv::KeyPoint> &keys, const std::vector<TDescriptor> &descriptors)
{
  DetectionResult result;

  pdetector->detectLoop(keys, descriptors, result);

  return result;
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void CustomLoopDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  double ts, x, y, t;
  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

#endif


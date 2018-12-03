/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include <DLoopDetector/DLoopDetector.h> // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <DLoopDetector/brief_extractor.h>
#include <DLoopDetector/orb_extractor.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

// static const char *VOC_FILE = "./resources/brief_k10L6.voc.gz";
// static const char *IMAGE_DIR = "./resources/images";
// static const char *POSE_FILE = "./resources/pose.txt";
static const char *VOC_FILE = "./pi_voc_orb.yml.gz";
static const char *IMAGE_DIR = "/Volumes/BILL_Mobile/png_converted/";
static const char *POSE_FILE = "./pos.txt";

static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;

static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";

typedef OrbVocabulary VOCTYPE;
typedef OrbLoopDetector LOOPDETCTORTYPE;
typedef FORB::TDescriptor DESCRIPTORTYPE;

// typedef BriefVocabulary VOCTYPE;
// typedef BriefLoopDetector LOOPDETCTORTYPE;
// typedef FBrief::TDescriptor DESCRIPTORTYPE;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main()
{
  std::cout << "demo demonstrating loop detection using DLoopDetector method" << std::endl;
  
  VOCTYPE voc(VOC_FILE);
  // Set loop detector parameters
  typename LOOPDETCTORTYPE::Parameters params(IMAGE_H, IMAGE_W);
  
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
  
  std::shared_ptr<LOOPDETCTORTYPE> pdetector = std::make_shared<LOOPDETCTORTYPE>(voc, params);

  try 
  {
    // run the demo with the given functor to extract features
    // BriefExtractor extractor(BRIEF_PATTERN_FILE);
    ORBExtractor extractor;

    // load image filenames  
    vector<string> filenames = 
      DUtils::FileFunctions::Dir(IMAGE_DIR, ".png", true);
    
    // prepare visualization windows
    DUtilsCV::GUI::tWinHandler win = "Current image";

    // prepare profiler to measure times
    DUtils::Profiler profiler;
    std::vector<cv::KeyPoint> keys(0);
    std::vector<DESCRIPTORTYPE> descriptors(0);

    int count = 0;
    
    // go
    for(unsigned int i = 0; i < filenames.size(); ++i)
    {
      // downsampling to 2Hz
      if(i % 15 != 0)
      {
        continue;
      }

      cout << "Adding image " << i << ": " << filenames[i] << "... " << endl;
      
      // get image
      cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale
      cv::Mat rgb;
      cvtColor(im, rgb, CV_GRAY2BGR); 

      // get features
      profiler.profile("features");
      extractor(im, keys, descriptors);
      profiler.stop();
          
      profiler.profile("detection");
      DetectionResult result;
      pdetector->detectLoop(keys, descriptors, result);   
      profiler.stop();          

      if(result.detection())
      {
        cout << "- Loop found with image " << result.match << "!"
          << endl;
        ++count;
      }
      else
      {
        cout << "- No loop: ";
        switch(result.status)
        {
          case CLOSE_MATCHES_ONLY:
            cout << "All the images in the database are very recent" << endl;
            break;
            
          case NO_DB_RESULTS:
            cout << "There are no matches against the database (few features in"
              " the image?)" << endl;
            break;
            
          case LOW_NSS_FACTOR:
            cout << "Little overlap between this image and the previous one"
              << endl;
            break;
              
          case LOW_SCORES:
            cout << "No match reaches the score threshold (alpha: " << ")" << endl;
            break;
            
          case NO_GROUPS:
            cout << "Not enough close matches to create groups. "
              << "Best candidate: " << result.match << endl;
            break;
            
          case NO_TEMPORAL_CONSISTENCY:
            cout << "No temporal consistency (k: " << "). "
              << "Best candidate: " << result.match << endl;
            break;
            
          case NO_GEOMETRICAL_CONSISTENCY:
            cout << "No geometrical consistency. Best candidate: " 
              << result.match << endl;
            break;
            
          default:
            break;
        }
      }
      
      cout << endl;
      
      if(result.detection())
      {
        std::string prompt = "Loop detected with image ";
        prompt += std::to_string(result.match);

        cv::putText(rgb, 
                    prompt.c_str(),
                    cv::Point(IMAGE_W/4,IMAGE_H/2), // Coordinates
                    cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                    1.0, // Scale. 2.0 = 2x bigger
                    cv::Scalar(0,0,255), // BGR Color
                    1, // Line Thickness (Optional)
                    8); // Anti-alias (Optional)         
      }
   
      // show image
      DUtilsCV::GUI::showImage(rgb, true, &win, 10);
    }

    if(count == 0)
    {
      cout << "No loops found in this image sequence" << endl;
    }
    else
    {
      cout << count << " loops found in this image sequence!" << endl;
    } 

    cout << endl << "Execution time:" << endl
      << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
      << " ms/image" << endl
      << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
      << " ms/image" << endl;

    cout << endl << "Press a key to finish..." << endl;

  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------


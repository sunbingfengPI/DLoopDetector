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

#include "demoDetector.h"
#include <DLoopDetector/brief_extractor.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

// static const char *VOC_FILE = "./resources/brief_k10L6.voc.gz";
// static const char *IMAGE_DIR = "./resources/images";
// static const char *POSE_FILE = "./resources/pose.txt";
static const char *VOC_FILE = "./pi_voc.yml.gz";
static const char *IMAGE_DIR = "/Volumes/BILL_Mobile/png_converted/";
static const char *POSE_FILE = "./pos.txt";

// static const int IMAGE_W = 640; // image size
// static const int IMAGE_H = 480;

static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main()
{
  // prepares the demo
  demoDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    BriefExtractor extractor(BRIEF_PATTERN_FILE);
    demo.run("BRIEF", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

